/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

import { dashboardVersionDefine } from './app-version'

// 测试必须跑在开发版 React 上（React.act 只存在于开发构建）。
// 宿主机若导出了 NODE_ENV=production，react-dom 会解析到生产构建，
// 全部组件测试都会以 "React.act is not a function" 失败，这里强制纠正。
process.env.NODE_ENV = 'test'

export default defineConfig({
  plugins: [react()],
  define: dashboardVersionDefine,
  test: {
    globals: true,
    environment: 'jsdom',
    // Node 22+ 自带实验性 localStorage 全局（未配 --localstorage-file 时为 undefined），
    // 会抢占 globalThis.localStorage 导致 jsdom 的实现不生效，这里显式关闭
    execArgv: ['--no-experimental-webstorage'],
    setupFiles: './src/test/setup.ts',
    // 每个用例前清空调用记录、重置 mock 实现、还原 vi.spyOn 打的桩，
    // 避免用例之间通过 mock 残留状态互相耦合（测试顺序变化就挂的隐性依赖）
    clearMocks: true,
    restoreMocks: true,
    mockReset: true,
    // 部分页面级集成测试要渲染整棵路由子树并等待多轮查询，默认 5s 偏紧
    testTimeout: 15000,
    hookTimeout: 15000,
    // 覆盖率：v8 provider，只统计 src 业务代码
    coverage: {
      provider: 'v8',
      // json-summary / json 便于脚本与 CI 程序化读取每文件覆盖率
      reporter: ['text', 'html', 'lcov', 'json-summary', 'json'],
      reportsDirectory: './coverage',
      include: ['src/**/*.{ts,tsx}'],
      exclude: [
        'src/**/*.test.{ts,tsx}',
        'src/**/__tests__/**',
        'src/test/**',
        'src/types/**',
        'src/**/*.d.ts',
        'src/main.tsx',
        'src/assets/**',
        // 产物类文件：Storybook 用例、代码生成产物、语言包资源，均无测试价值
        'src/**/*.stories.{ts,tsx}',
        'src/**/*.gen.ts',
        'src/i18n/locales/**',
      ],
      // 防退化闸门：阈值取当前实测基线略低一档，只用于拦住覆盖率下滑，
      // 不作为质量目标。随着测试补充应逐步上调这些数字。
      thresholds: {
        lines: 73,
        statements: 72,
        functions: 68,
        branches: 58,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
