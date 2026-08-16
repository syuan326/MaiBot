/**
 * MaiSaka 监控页面入口
 *
 * 通过 WebSocket 实时渲染 MaiSaka 推理过程。
 */
import { AuthMonitor } from './auth-monitor'
import { MaisakaMonitor } from './maisaka-monitor'

export function PlannerMonitorPage() {
  return (
    <div className="min-w-0 max-w-full space-y-4 overflow-x-hidden px-4 pb-4 pt-2 sm:space-y-6 sm:px-6 sm:pb-6 sm:pt-3">
      {/* 主体 */}
      <MaisakaMonitor />
    </div>
  )
}

export function AuthMonitorPage() {
  return (
    <div className="min-w-0 max-w-full space-y-4 overflow-x-hidden px-4 pb-4 pt-2 sm:space-y-6 sm:px-6 sm:pb-6 sm:pt-3">
      {/* 主体 */}
      <AuthMonitor />
    </div>
  )
}
