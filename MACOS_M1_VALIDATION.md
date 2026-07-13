# VisionForge Apple Silicon 검증 기록

> 검증일: 2026-07-13
> 범위: macOS arm64 개발 환경, MPS 기능 경로, PyInstaller sidecar, Tauri app·dmg

## 검증 장비

| 항목 | 값 |
|---|---|
| 장비 | MacBook Pro, Apple M1 Pro |
| CPU·메모리 | 10 cores, 32GB unified memory |
| 운영체제 | macOS 26.5.1, arm64 |
| Node·npm | 24.18.0, 11.16.0 |
| Rust | 1.97.0, `aarch64-apple-darwin` |
| Python·uv | 3.12.13, uv 0.11.28 |
| PyTorch·TorchVision | 2.11.0, 0.26.0 |

이 장비는 최종 최소 기준인 M1 Pro 16GB보다 메모리가 많다. 따라서 기능 호환성 근거로는 사용할 수 있지만 16GB 메모리 한계나 72시간 완료 기준의 수용 증거는 아니다.

## 결과

- Rust workspace: core 11 passed, desktop 3 passed
- Python 기본 테스트: 14 passed, Torch opt-in 1 skipped
- Ruff: passed
- TypeScript·Vitest: passed, 4 tests
- Vite production build: passed
- 사전학습 Faster R-CNN MPS 1 epoch 학습·평가·추론: passed
- `PYTORCH_ENABLE_MPS_FALLBACK=0`: passed
- 기본 M1 정책 640–960px, batch 1, accumulation 8, backbone 2층: passed
- 2000×2000 전체 이미지와 순차 타일 MPS 추론: passed
- 동일 저장 모델의 MPS·CPU 소형 fixture 결과: Box와 직렬화 점수 일치
- sidecar `system-profile`: `APPLE_HIGH_MEMORY`, providers `mps,cpu`
- MPS 권장 메모리 한도: 26,800,603,136 bytes

통합 fixture는 5장의 단순 도형과 1 epoch만 사용한다. 위 결과는 실행 경로 검증이며 탐지 품질이나 실사용 처리량의 증거가 아니다.

## 번들 결과

| 산출물 | 결과 |
|---|---|
| PyInstaller sidecar | arm64, 약 537MB, 번들 가중치 SHA-256 일치 |
| `VisionForge.app` | arm64, 약 815MB, ad-hoc hardened-runtime 서명 검증 통과 |
| `VisionForge_0.1.0_aarch64.dmg` | 약 272MB, `hdiutil verify` 통과 |
| DMG 내용 | `VisionForge.app`, `/Applications` 링크, 마운트 후 앱 서명 검증 통과 |

검증 시 DMG SHA-256은 `7a63eee505bb3b19265a1debb398aaa8d9b2839e89db410de945d0d09cb537f6`였다. 다시 빌드하면 서명·메타데이터 때문에 값이 바뀔 수 있다.

## 이번에 수정한 macOS 결함

1. Node 복사 과정에서 PyInstaller의 상대 symlink가 삭제 예정 임시 폴더의 절대 경로로 변환되던 문제를 `verbatimSymlinks`로 수정했다.
2. sidecar 복사 직후 번들 실행 파일로 `system-profile` smoke test를 수행해 깨진 런타임을 빌드 단계에서 차단한다.
3. PyInstaller target을 실행 중인 arm64 Python·Node 아키텍처와 대조하고 `--target-arch arm64`를 명시한다.
4. release 앱에서 빌드 머신의 개발 `.venv`로 fallback하지 못하게 해 누락된 sidecar를 숨기지 않는다.
5. 읽지 못한 Apple 메모리를 고성능 프로필로 오판하지 않으며, MPS 메모리는 전체 RAM 대신 PyTorch 권장 한도를 보고한다.
6. DMG 빌드는 `--ci`로 Finder 자동화 의존성을 제거하고 로컬 app은 ad-hoc identity로 서명한다.

## 남은 수용 조건

- M1 Pro 16GB에서 동일 시험과 실제 데이터 메모리 한계 측정
- 24시간 이상 학습, 절전·강제 종료·체크포인트 재개, 72시간 기준 작업
- 실제 촬영 고정 평가 세트와 모델 승격 기준
- MPS OOM 후 다음 항목 격리·cache 정리·복구 시험
- macOS 14 실제 장비 또는 macOS 14 빌드 호스트의 하위 호환성 검증
- Developer ID Application 서명, notarization, stapling, Gatekeeper 최초 실행
- 출시 전 고유 Bundle ID 확정과 사전학습 모델 라이선스 검토
