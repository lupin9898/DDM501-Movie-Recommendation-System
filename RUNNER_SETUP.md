# Self-Hosted Runner Setup

Hướng dẫn cài đặt GitHub Actions self-hosted runner để chạy các workflow `deploy` và `train`.

---

## Kiến trúc CI/CD

```
Push to main
     │
     ▼
┌─────────────────────────────┐
│  CI  (GitHub-hosted)        │
│  lint → test → build → push │
└──────────────┬──────────────┘
               │ workflow_run (success)
               ▼
┌─────────────────────────────┐
│  Deploy  (self-hosted)      │
│  pull image → compose up    │
│  health check → rollback    │
└─────────────────────────────┘

Manual / Weekly schedule
     │
     ▼
┌─────────────────────────────┐
│  Train  (self-hosted)       │
│  preprocess → train →       │
│  MLflow log → reload API    │
└─────────────────────────────┘
```

---

## Yêu cầu máy chủ

| Item | Yêu cầu tối thiểu |
|------|-------------------|
| OS | Ubuntu 22.04 LTS / macOS 13+ |
| CPU | 4 cores |
| RAM | 8 GB (16 GB để train ALS trên full ML-25M) |
| Disk | 50 GB (data + Docker images + MLflow artifacts) |
| Network | Truy cập được GitHub + ghcr.io |
| Docker | Docker Engine 24+ với Compose plugin |

---

## Cài đặt nhanh

### Bước 1 — Lấy Registration Token

Truy cập: **Repository → Settings → Actions → Runners → New self-hosted runner**

Copy token hiển thị (có hiệu lực 1 giờ).

### Bước 2 — Chạy script cài đặt

```bash
# Clone repo về máy chủ (nếu chưa có)
git clone https://github.com/OWNER/REPO.git
cd REPO

# Cấp quyền thực thi
chmod +x scripts/setup_runner.sh

# Cài đặt runner
./scripts/setup_runner.sh \
  --repo  https://github.com/OWNER/REPO \
  --token <REGISTRATION_TOKEN>
```

Script sẽ tự động:
- Cài Docker, git (Linux)
- Tải runner binary phù hợp với OS/arch
- Cấu hình và đăng ký với repo
- Cài service (systemd / launchd)

### Bước 3 — Xác nhận runner hoạt động

Vào **Settings → Actions → Runners** — runner phải hiển thị trạng thái **Idle**.

---

## Cài đặt môi trường production

### 1. Tạo GitHub Environment

**Settings → Environments → New environment** → đặt tên `production`

Thêm protection rules nếu cần (require reviewers, deployment branches).

### 2. GitHub Secrets (tùy chọn)

| Secret | Mô tả |
|--------|-------|
| `GRAFANA_PASSWORD` | Mật khẩu admin Grafana (mặc định: `admin`) |

### 3. Chuẩn bị thư mục trên host

```bash
# Thư mục data (cần có trước khi chạy train)
mkdir -p data/raw data/processed

# Thư mục artifacts (model.pkl sẽ được lưu ở đây)
mkdir -p artifacts

# Thư mục monitoring (prometheus config)
# Đã có trong repo — không cần tạo thêm
```

---

## Quản lý service

### Linux (systemd)

```bash
cd ~/actions-runner

sudo ./svc.sh status    # kiểm tra trạng thái
sudo ./svc.sh stop      # dừng
sudo ./svc.sh start     # khởi động
sudo ./svc.sh uninstall # gỡ cài đặt
```

Xem logs:
```bash
journalctl -u actions.runner.*.service -f
```

### macOS (launchd)

```bash
cd ~/actions-runner

./svc.sh status
./svc.sh stop
./svc.sh start
./svc.sh uninstall
```

---

## Workflows

### CI (`ci.yml`) — GitHub-hosted
Tự động chạy khi push/PR. Không cần runner riêng.

```
lint (ruff) → test (pytest --cov) → build (Docker) → push (ghcr.io)
```

### Deploy (`deploy.yml`) — self-hosted
Tự động trigger sau khi CI pass trên `main`, hoặc chạy thủ công.

```
Actions → Deploy → Run workflow → chọn image tag (mặc định: latest)
```

### Train (`train.yml`) — self-hosted
Chạy thủ công hoặc tự động mỗi Chủ nhật 02:00 UTC.

```
Actions → Train → Run workflow → chọn model type
```

---

## Cập nhật runner

```bash
cd ~/actions-runner

# Dừng service
sudo ./svc.sh stop      # Linux
./svc.sh stop           # macOS

# Chạy lại script với version mới (sửa RUNNER_VERSION trong script)
./scripts/setup_runner.sh --repo <URL> --token <NEW_TOKEN>
```

---

## Troubleshooting

**Runner offline sau reboot**
```bash
sudo systemctl enable actions.runner.*.service  # Linux — bật autostart
```

**Docker permission denied**
```bash
sudo usermod -aG docker $USER
# Đăng xuất và đăng nhập lại
```

**Port 8000 đã bị dùng**
```bash
sudo lsof -i :8000
# Dừng process đang dùng port hoặc đổi port trong docker-compose.prod.yml
```

**Health check fail sau deploy**
```bash
docker compose -f docker-compose.prod.yml logs api --tail=100
```
