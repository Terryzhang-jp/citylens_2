# CityLens 部署指南

## 线上地址

| 服务 | URL |
|------|-----|
| 前端 | https://frontend-chi-fawn-52.vercel.app |
| 后端 API | https://citylens-api-1083982545507.asia-northeast1.run.app |
| GitHub | https://github.com/Terryzhang-jp/citylens_2 |

---

## 前置要求

### CLI 工具
```bash
# Google Cloud CLI
brew install google-cloud-sdk

# Vercel CLI
npm i -g vercel
```

### 登录认证
```bash
# Google Cloud
gcloud auth login
gcloud config set project gxutokyo

# Vercel
vercel login
```

---

## 后端部署 (Google Cloud Run)

### 快速部署
```bash
cd backend

gcloud run deploy citylens-api \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars="GEMINI_API_KEY=你的API密钥"
```

### 部署参数说明
| 参数 | 说明 |
|------|------|
| `--source .` | 使用当前目录的 Dockerfile 构建 |
| `--region asia-northeast1` | 部署区域（东京） |
| `--allow-unauthenticated` | 允许公开访问 |
| `--memory 2Gi` | 内存配置（MobileSAM 需要较大内存） |
| `--timeout 300` | 请求超时时间（秒） |

### 仅更新环境变量
```bash
gcloud run services update citylens-api \
  --region asia-northeast1 \
  --set-env-vars="GEMINI_API_KEY=新的API密钥"
```

### 查看日志
```bash
gcloud run services logs read citylens-api --region asia-northeast1 --limit 50
```

### 查看服务状态
```bash
gcloud run services describe citylens-api --region asia-northeast1
```

---

## 前端部署 (Vercel)

### 快速部署
```bash
cd frontend

# 部署到生产环境
vercel --prod
```

### 更新环境变量
```bash
# 查看当前环境变量
vercel env ls

# 添加/更新环境变量
echo "https://citylens-api-xxx.run.app" | vercel env add NEXT_PUBLIC_API_URL production

# 删除环境变量
vercel env rm NEXT_PUBLIC_API_URL production
```

### 重新部署（环境变量更新后需要）
```bash
vercel --prod
```

### 查看部署状态
```bash
vercel ls
```

### 查看部署日志
```bash
vercel logs https://frontend-xxx.vercel.app
```

---

## 完整重新部署流程

### 1. 更新代码并推送
```bash
git add -A
git commit -m "your changes"
git push
```

### 2. 部署后端
```bash
cd backend
gcloud run deploy citylens-api \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars="GEMINI_API_KEY=你的API密钥"
```

### 3. 部署前端
```bash
cd frontend
vercel --prod
```

---

## 配置文件说明

### 后端

| 文件 | 用途 |
|------|------|
| `Dockerfile` | Docker 构建配置 |
| `.dockerignore` | Docker 构建时忽略的文件 |
| `.gcloudignore` | gcloud 上传时忽略的文件（重要：排除 venv/） |
| `.env` | 本地环境变量（不会上传） |
| `.env.example` | 环境变量模板 |

### 前端

| 文件 | 用途 |
|------|------|
| `.env.local` | 本地环境变量 |
| `vercel.json` | Vercel 配置（如有） |

---

## 环境变量

### 后端 (Cloud Run)
| 变量名 | 说明 |
|--------|------|
| `GEMINI_API_KEY` | Google Gemini API 密钥 |

### 前端 (Vercel)
| 变量名 | 说明 |
|--------|------|
| `NEXT_PUBLIC_API_URL` | 后端 API 地址 |

---

## 常见问题

### Q: 部署后端时上传文件太大/太慢
确保 `.gcloudignore` 文件存在并包含：
```
venv/
__pycache__/
*.pt
.env
```

### Q: 前端无法连接后端 API
1. 检查后端 CORS 配置（`main.py`）是否包含前端域名
2. 检查前端环境变量 `NEXT_PUBLIC_API_URL` 是否正确
3. 更新后需要重新部署前端

### Q: MobileSAM 模型加载失败
模型会在首次请求时自动下载（~24MB）。如果下载失败，会自动回退到简单裁剪模式。

### Q: Cloud Run 冷启动慢
可以配置最小实例数（会产生费用）：
```bash
gcloud run services update citylens-api \
  --region asia-northeast1 \
  --min-instances 1
```

---

## 费用说明

### Google Cloud Run
- 免费额度：每月 200 万次请求、360,000 GB-秒
- 超出后按量计费

### Vercel
- Hobby 计划免费
- 包含自定义域名、HTTPS

---

## 回滚

### 后端回滚
```bash
# 查看历史版本
gcloud run revisions list --service citylens-api --region asia-northeast1

# 回滚到指定版本
gcloud run services update-traffic citylens-api \
  --region asia-northeast1 \
  --to-revisions=citylens-api-00001-xxx=100
```

### 前端回滚
```bash
# 查看历史部署
vercel ls

# 重新部署指定版本
vercel rollback [deployment-url]
```
