import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxy = {
  "/api": {
    target: "http://localhost:8000",
    changeOrigin: true,
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // 개발 서버(npm run dev): HMR 웹소켓이 있어, 백그라운드 탭에서 연결이 끊기면 돌아올 때 페이지가 새로고침된다.
  server: {
    host: true,
    port: 3010,
    proxy: apiProxy,
  },
  // 빌드 결과 서빙(npm start): 웹소켓이 없어 탭을 오가도 새로고침되지 않는다.
  preview: {
    host: true,
    port: 3010,
    proxy: apiProxy,
  },
})
