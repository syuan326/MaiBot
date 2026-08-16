/**
 * 麦麦视察 - 鉴权组件日志实时监控
 *
 * 通过 WebSocket 实时接收鉴权事件（auth.result / auth.input_injection），
 * 以时间线形式展示鉴权器的工作日志：Planner/Replyer 身份核对结果、
 * 输入注入检测命中与确认方式等。
 */
import {
  AlertTriangle,
  ChevronDown,
  Eraser,
  Fingerprint,
  ShieldAlert,
  ShieldCheck,
  ShieldX,
  XCircle,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'

import type {
  AuthInputInjectionEvent,
  AuthResultEvent,
} from '@/lib/maisaka-monitor-client'
import type { TimelineEntry } from './use-maisaka-monitor'
import { useMaisakaMonitor } from './use-maisaka-monitor'

// ─── 工具函数 ──────────────────────────────────────────────────

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 10) return '刚刚'
  if (diff < 60) return `${Math.round(diff)}秒前`
  if (diff < 3600) return `${Math.round(diff / 60)}分钟前`
  return `${Math.round(diff / 3600)}小时前`
}

function resolveSessionName(sessions: ReturnType<typeof useMaisakaMonitor>['sessions'], sessionId: string): string {
  return sessions.get(sessionId)?.sessionName ?? sessionId.slice(0, 8)
}

function getStageLabel(stage: string): string {
  if (stage === 'planner') return 'Planner 决策'
  if (stage === 'replyer') return 'Replyer 回复'
  return stage || '未知阶段'
}

function getConfirmMethodLabel(method: string): string {
  if (method === 'rule') return '规则'
  if (method === 'llm') return 'LLM'
  if (method === 'rule_then_llm') return '规则+LLM'
  return method || '未知'
}

const ISSUE_TYPE_LABELS: Record<string, string> = {
  wrong_attribution: '归属错误',
  wrong_target: '对象错误',
  wrong_name: '称呼混淆',
  self_confusion: '自我混淆',
  identity_conflict: '身份矛盾',
  relation_conflict: '关系矛盾',
  prompt_leak: '提示词泄露',
  personality_violation: '人格违反',
  content_violation: '敏感内容',
  injection_propagation: '注入传播',
}

function getIssueTypeLabel(issueType: string): string {
  return ISSUE_TYPE_LABELS[issueType] ?? issueType
}

// ─── 鉴权结果卡片 ──────────────────────────────────────────────

function AuthResultCard({
  data,
  sessionName,
}: {
  data: AuthResultEvent
  sessionName: string
}) {
  const isRejected = !data.passed
  const borderClassName = data.audit_error
    ? 'border-amber-500/35'
    : isRejected
      ? 'border-red-500/35'
      : 'border-emerald-500/35'

  const identityCheck = data.identity_check as
    | { is_target?: unknown; summary?: string; forbidden_names?: string[] }
    | null
    | undefined
  const identitySummary = identityCheck?.summary?.trim()

  return (
    <div className={cn('rounded-md border px-3 py-2.5', borderClassName)}>
      <div className="flex flex-wrap items-center gap-2">
        {data.audit_error ? (
          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-500" />
        ) : isRejected ? (
          <ShieldX className="h-4 w-4 shrink-0 text-red-500" />
        ) : (
          <ShieldCheck className="h-4 w-4 shrink-0 text-emerald-500" />
        )}
        <span className="text-sm font-medium">
          {data.audit_error ? '审核异常放行' : isRejected ? '鉴权驳回' : '鉴权通过'}
        </span>
        <Badge variant="outline" className="text-[10px]">
          {getStageLabel(data.stage)}
        </Badge>
        {data.cycle_id != null && (
          <Badge variant="secondary" className="text-[10px]">
            #{data.cycle_id}
          </Badge>
        )}
        {isRejected && !data.audit_error && (
          <Badge variant="destructive" className="text-[10px]">
            尝试 {data.attempt}/{Math.max(data.max_retries, 1)}
          </Badge>
        )}
        <span className="text-muted-foreground ml-auto flex shrink-0 items-center gap-1.5 text-[11px]">
          <span className="max-w-40 truncate" title={sessionName}>
            {sessionName}
          </span>
          <span>{formatTimestamp(data.timestamp)}</span>
          <span className="text-muted-foreground/70">{formatRelativeTime(data.timestamp)}</span>
        </span>
      </div>

      {identitySummary && (
        <div className="mt-1.5 flex items-start gap-1.5 text-xs text-muted-foreground">
          <Fingerprint className="mt-0.5 h-3 w-3 shrink-0" />
          <span className="min-w-0 break-words">{identitySummary}</span>
        </div>
      )}

      {data.audit_error && data.error && (
        <div className="mt-1.5 rounded-md border border-amber-500/30 bg-amber-500/5 px-2.5 py-1.5 text-xs">
          <span className="mb-0.5 block font-medium text-amber-600 dark:text-amber-400">
            失败原因
          </span>
          <p className="min-w-0 break-words font-mono text-[11px] leading-4">{data.error}</p>
        </div>
      )}

      {data.reason && (
        <p className="text-foreground/85 mt-1.5 text-sm break-words whitespace-pre-wrap">
          {data.reason}
        </p>
      )}

      {data.issues.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {data.issues.map((issue, index) => (
            <li
              key={`${issue.issue_type}-${index}`}
              className="text-muted-foreground flex items-start gap-1.5 text-xs"
            >
              <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-red-500" />
              <span className="shrink-0 font-medium">
                {getIssueTypeLabel(issue.issue_type)}
              </span>
              <span className="min-w-0 break-words">{issue.detail}</span>
            </li>
          ))}
        </ul>
      )}

      {data.rejected_text && (
        <div className="bg-muted/40 text-muted-foreground mt-2 rounded-md border px-2.5 py-1.5 text-xs break-words whitespace-pre-wrap">
          <span className="mb-0.5 block text-[10px] font-medium">被驳回内容</span>
          {data.rejected_text}
        </div>
      )}
    </div>
  )
}

// ─── 输入注入检测卡片 ──────────────────────────────────────────

function AuthInjectionCard({
  data,
  sessionName,
}: {
  data: AuthInputInjectionEvent
  sessionName: string
}) {
  return (
    <div className="rounded-md border border-red-500/40 bg-red-500/5 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <ShieldAlert className="h-4 w-4 shrink-0 text-red-500" />
        <span className="text-sm font-medium">检测到注入攻击</span>
        <Badge variant="destructive" className="text-[10px]">
          命中 {data.hit_count} 条规则
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {getConfirmMethodLabel(data.confirm_method)}确认
        </Badge>
        <span className="text-muted-foreground ml-auto flex shrink-0 items-center gap-1.5 text-[11px]">
          <span className="max-w-40 truncate" title={sessionName}>
            {sessionName}
          </span>
          <span>{formatTimestamp(data.timestamp)}</span>
          <span className="text-muted-foreground/70">{formatRelativeTime(data.timestamp)}</span>
        </span>
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs">
        <span className="text-foreground/80 font-medium">
          {data.user_name || '未知用户'}
          {data.user_id && <span className="text-muted-foreground/70">（{data.user_id}）</span>}
        </span>
        {data.categories.map((category) => (
          <Badge key={category} variant="secondary" className="text-[10px]">
            {category}
          </Badge>
        ))}
      </div>

      {data.text && (
        <p className="text-foreground/85 mt-1.5 text-sm break-words whitespace-pre-wrap">
          {data.text}
        </p>
      )}

      {data.reason && (
        <p className="text-muted-foreground mt-1 text-xs break-words whitespace-pre-wrap">
          判定理由：{data.reason}
        </p>
      )}
    </div>
  )
}

// ─── 时间线条目渲染器 ──────────────────────────────────────────

function AuthTimelineRenderer({
  entry,
  sessionName,
}: {
  entry: TimelineEntry
  sessionName: string
}) {
  if (entry.type === 'auth.result') {
    return <AuthResultCard data={entry.data as AuthResultEvent} sessionName={sessionName} />
  }
  if (entry.type === 'auth.input_injection') {
    return (
      <AuthInjectionCard
        data={entry.data as AuthInputInjectionEvent}
        sessionName={sessionName}
      />
    )
  }
  return null
}

// ─── 主组件 ─────────────────────────────────────────────────

type AuthFilterMode = 'all' | 'abnormal' | 'passed'

export function AuthMonitor() {
  const { timeline, sessions, connected, clearTimeline } = useMaisakaMonitor()
  const scrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [filterMode, setFilterMode] = useState<AuthFilterMode>('all')

  const authEntries = useMemo(
    () =>
      timeline.filter(
        (entry) => entry.type === 'auth.result' || entry.type === 'auth.input_injection'
      ),
    [timeline]
  )

  const visibleEntries = useMemo(() => {
    if (filterMode === 'all') return authEntries
    if (filterMode === 'passed') {
      return authEntries.filter((entry) => {
        if (entry.type !== 'auth.result') return false
        const data = entry.data as AuthResultEvent
        return data.passed && !data.audit_error
      })
    }
    return authEntries.filter((entry) => {
      if (entry.type === 'auth.input_injection') return true
      const data = entry.data as AuthResultEvent
      return !data.passed || data.audit_error
    })
  }, [authEntries, filterMode])

  const stats = useMemo(() => {
    let passedCount = 0
    let rejectedCount = 0
    let injectionCount = 0
    for (const entry of authEntries) {
      if (entry.type === 'auth.input_injection') {
        injectionCount += 1
        continue
      }
      const data = entry.data as AuthResultEvent
      if (data.passed) passedCount += 1
      else rejectedCount += 1
    }
    return { passed: passedCount, rejected: rejectedCount, injections: injectionCount }
  }, [authEntries])

  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = 'smooth') => {
      const viewport = scrollRef.current?.querySelector(
        '[data-radix-scroll-area-viewport]'
      ) as HTMLDivElement | null
      viewport?.scrollTo({ top: viewport.scrollHeight, behavior })
      setAutoScroll(true)
    },
    []
  )

  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const target = e.currentTarget as HTMLElement
      const { scrollTop, scrollHeight, clientHeight } = target
      setAutoScroll(scrollHeight - scrollTop - clientHeight < 80)
    },
    []
  )

  useEffect(() => {
    if (autoScroll) {
      requestAnimationFrame(() => scrollToBottom('auto'))
    }
  }, [visibleEntries.length, autoScroll, scrollToBottom])

  const filterTabs: { mode: AuthFilterMode; label: string; count: number }[] = [
    { mode: 'all', label: '全部', count: authEntries.length },
    { mode: 'abnormal', label: '仅异常', count: stats.rejected + stats.injections },
    { mode: 'passed', label: '仅通过', count: stats.passed },
  ]

  return (
    <div className="flex min-w-0 flex-col gap-3">
      {/* 统计与操作条 */}
      <div className="bg-background flex min-w-0 flex-wrap items-center gap-2 rounded-md border px-2 py-1.5">
        <div className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />
          <span>通过 {stats.passed}</span>
          <span className="text-muted-foreground/40">|</span>
          <ShieldX className="h-3.5 w-3.5 text-red-500" />
          <span>驳回 {stats.rejected}</span>
          <span className="text-muted-foreground/40">|</span>
          <ShieldAlert className="h-3.5 w-3.5 text-red-500" />
          <span>注入 {stats.injections}</span>
          {connected && (
            <span className="bg-emerald-500 ml-1 h-2 w-2 rounded-full" title="已连接" />
          )}
        </div>

        <div className="flex shrink-0 items-center gap-1">
          {filterTabs.map((tab) => (
            <button
              key={tab.mode}
              type="button"
              onClick={() => setFilterMode(tab.mode)}
              className={cn(
                'rounded-md px-2 py-0.5 text-xs transition-colors',
                filterMode === tab.mode
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
              )}
            >
              {tab.label}
              <span className="ml-1 opacity-60">{tab.count}</span>
            </button>
          ))}
        </div>

        <div className="ml-auto flex shrink-0 items-center gap-1.5">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 shrink-0 px-2 text-[11px]"
            onClick={() => scrollToBottom('smooth')}
            title="回到底部"
          >
            <ChevronDown className={cn('mr-1 h-3 w-3', autoScroll && 'text-primary')} />
            回到底部
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0"
            onClick={clearTimeline}
            title="清空"
            aria-label="清空"
          >
            <Eraser className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* 鉴权日志时间线 */}
      <Card className="min-h-[420px] min-w-0 flex-1 overflow-hidden">
        <ScrollArea className="h-[calc(100vh-220px)]" ref={scrollRef} onScrollCapture={handleScroll}>
          <div className="min-w-0 space-y-2 p-4">
            {visibleEntries.length === 0 ? (
              <div className="text-muted-foreground flex flex-col items-center justify-center gap-3 py-20">
                <ShieldCheck className="h-10 w-10 opacity-30" />
                <p className="text-sm">等待鉴权事件…</p>
                <p className="max-w-md text-center text-xs opacity-60">
                  当鉴权器完成身份核对或检测到输入注入时，日志会实时展示在这里
                </p>
              </div>
            ) : (
              visibleEntries.map((entry) => (
                <div key={entry.id} className="min-w-0">
                  <AuthTimelineRenderer
                    entry={entry}
                    sessionName={resolveSessionName(sessions, entry.sessionId)}
                  />
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </Card>
    </div>
  )
}
