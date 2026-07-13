# VisionForge

VisionForge는 대상 이미지, 실제 촬영 상황 설명, 결과 폴더 규칙으로 동적 작업 명세를 만들고 로컬 학습 데이터를 생성한 뒤 실제 이미지 세트를 보수적으로 분류하는 오프라인 데스크톱 앱입니다.

현재 저장소에는 Windows에서 검증한 기능형 MVP와 PyTorch 고성능 탐지 백엔드가 구현되어 있으며, M1 Pro 32GB에서 macOS arm64 기능 검증도 완료했습니다. 클라우드 API나 로컬 LLM을 사용하지 않으며, 객체 탐지는 TorchVision 모델을 로컬에서 미세조정합니다. 배포 앱은 Python·PyTorch 엔진을 분리된 sidecar 런타임으로 포함합니다.

최종 필수 실행 기준은 MacBook Pro 16-inch M1 Pro 16GB이며, 지원 작업의 로컬 학습·평가·추론 전체 흐름을 최대 72시간 안에 완료하도록 설계합니다. MPS 장치 선택, M1 전용 batch·gradient accumulation, arm64 sidecar와 app·dmg 빌드는 32GB 장비에서 기능 검증됐습니다. 16GB 기준 장비의 장시간 안정성·72시간 성능 시험은 아직 필요합니다. Core ML·ONNX 경로는 현재 구현 범위가 아닙니다.

## 구현된 흐름

1. 프로젝트 생성 또는 기존 프로젝트 복원
2. 상황 설명, 포함·미포함·검토·실패 폴더와 이중 임계값을 TaskSpec 리비전으로 저장
3. 대상·배경 이미지 다중 등록, 손상·중복·블러·밝기 검사
4. 상황 키워드를 크기·회전·조명·블러·노이즈·가림 분포로 컴파일해 재현 가능한 2D 합성
5. 합성본 승인·제외, 최종 가시 마스크 기반 Box 생성과 좌표 수정
6. 체크섬·Box·그룹 누수·TaskSpec 계보를 거친 불변 데이터셋 버전 생성
7. Faster R-CNN MobileNetV3 FPN 사전학습 모델 미세조정, 고정 분할 평가, 조기 종료와 체크포인트 재개
8. MPS·CUDA·CPU 장치 자동 선택, 실제 이미지 순차·고해상도 타일 추론, NMS와 항목별 실패 격리
9. 원본을 보존하며 사용자 지정 포함·미포함·검토·실패 폴더로 복사하고 JSON manifest 생성
10. TaskSpec을 포함한 `.vfmodel` 내보내기·가져오기와 안전 검증
11. Windows·macOS별 Tauri 번들 설정과 PyInstaller `onedir` sidecar 리소스 생성

## 저장 정책

- 영구 저장: 등록 원본, 검토할 합성본, 승인 상태, 불변 데이터셋, 모델, 추론 결과
- 제한 저장: 빌드 캐시와 재생성 가능한 중간물
- 메모리 전용: 학습 시점의 텐서 변형과 배치. 처리 후 즉시 해제하며 변형본을 별도 이미지 파일로 쌓지 않음
- 모든 프로젝트 데이터는 사용자가 선택한 로컬 폴더에 저장

합성 데이터와 실제 사진의 차이는 실제 환경과 가까운 배경, 부정 이미지, 그룹 분할, 실제 사진 고정 평가, 오탐·미탐 재학습 후보 흐름으로 줄이는 구조입니다. 모델은 실제 사진 품질 게이트를 통과하기 전 `candidate`로 저장되며, 후보 모델의 결과는 점수가 높아도 자동 폴더가 아니라 검토 폴더로만 전달됩니다.

## 프로젝트 폴더 사용법

- 새 프로젝트는 이름과 대상 클래스를 입력하고 `상위 저장 폴더를 선택하고 새 프로젝트 만들기`를 누릅니다. 선택한 위치 아래에 `vf-...` 전용 폴더가 자동 생성되므로 빈 프로젝트 폴더를 미리 만들 필요가 없습니다.
- 기존 프로젝트 열기는 `project.json`과 `project.sqlite`가 들어 있는 정확한 `vf-...` 폴더를 선택할 때 사용합니다. 빈 폴더나 프로젝트들의 상위 폴더는 기존 프로젝트가 아닙니다.
- 작업 화면의 `프로젝트 선택 화면` 버튼으로 첫 화면에 돌아갈 수 있습니다. 다른 프로젝트를 열기 전까지 현재 작업공간은 유지되며 `현재 프로젝트로 돌아가기`로 복귀할 수 있습니다.

## 구조

```text
apps/desktop/                 React 19 + Tauri 2 데스크톱 UI
apps/desktop/src-tauri/       Rust 명령 계층과 Windows·macOS 번들 설정
crates/visionforge-core/      SQLite 프로젝트·자산·작업·데이터셋·모델 도메인
engine/                       Pillow/NumPy 이미지·학습·추론·패키지 엔진
scripts/                      sidecar 및 설치 번들 재현 스크립트
VISIONFORGE_PRODUCT_PLAN.md   전체 제품 기획과 시스템 구조 기준선
IMPLEMENTATION_STATUS.md      구현 범위, 검증 결과, 후속 위험
VISIONFORGE_REQUIREMENTS_SPEC.md 통합 요구사항과 수용 기준
HARDWARE_REQUIREMENTS.md      학습·추론 장비 등급과 구매 전 벤치마크 기준
HIGH_PERFORMANCE_BACKEND_IMPLEMENTATION.md 고성능 백엔드 구현·검증 상세
MACOS_M1_VALIDATION.md        Apple Silicon 실기 검증 결과와 잔여 조건
```

## 개발

필수 도구는 Node.js 24, Rust 1.85 이상, Python 3.12와 uv입니다.

```bash
uv sync --project engine --all-groups --locked
npm ci
cargo test --workspace --locked
uv run --project engine pytest engine/tests -p no:cacheprovider
npm --workspace @visionforge/desktop run check
```

개발 앱 실행:

```powershell
npm --workspace @visionforge/desktop run tauri -- dev
```

현재 운영체제용 sidecar와 설치 번들 생성:

```text
npm --workspace @visionforge/desktop run build:sidecar
npm --workspace @visionforge/desktop run tauri -- build
```

Windows는 NSIS, macOS는 `.app`과 `.dmg`를 생성하도록 플랫폼별 설정을 사용합니다. macOS 배포 서명과 notarization에는 별도 Apple 인증 정보가 필요합니다.

### Apple Silicon macOS

M1 계열 Mac에서는 arm64 Node·Python으로 환경을 만들고 arm64 번들을 명시적으로 생성합니다.

```bash
brew install node@24 python@3.12 rustup uv
export PATH="/opt/homebrew/opt/node@24/bin:/opt/homebrew/opt/rustup/bin:$PATH"
rustup default stable
uv sync --python /opt/homebrew/bin/python3.12 --project engine --all-groups --locked
npm ci
PYTORCH_ENABLE_MPS_FALLBACK=0 VISIONFORGE_DEVICE=mps \
  VISIONFORGE_RUN_TORCH_INTEGRATION=1 VISIONFORGE_TEST_PRETRAINED=1 \
  uv run --project engine pytest engine/tests/test_torch_detector.py -p no:cacheprovider
npm run build:macos
```

`build:macos`는 로컬 실행용 ad-hoc 서명 번들을 만듭니다. 외부 배포에는 Developer ID Application 인증서와 notarization 자격 증명이 필요하며, 이때 `APPLE_SIGNING_IDENTITY`를 설정하고 `npm run build:macos:signed`를 실행합니다. 설정한 identity는 PyInstaller sidecar의 Python·PyTorch 바이너리에도 전달됩니다.

## 현재 한계

- PyTorch 탐지 엔진은 구현됐지만 실제 촬영 고정 평가 세트가 없어 상용 정확도를 입증하지 못했습니다.
- 범용 동적 v1은 `object_presence` 작업만 실행합니다. OCR, 세밀 분류, 이상 탐지는 지원 백엔드가 추가되기 전까지 실행하지 않습니다.
- 자연어 처리는 현재 규칙·키워드 컴파일러입니다. 지원하지 않는 원근·다중 객체 조건은 경고하고 임의 구현하지 않습니다.
- 체크포인트 재개는 구현됐지만 실제 M1에서 절전·강제 종료·24시간 이상 학습 복구 시험이 필요합니다.
- macOS arm64 sidecar·MPS·ad-hoc app·dmg는 M1 Pro 32GB에서 검증됐지만, Developer ID 서명·notarization과 macOS 14 실기 호환성은 미완료입니다.
- Core ML·ONNX 추론, 실제 평가 세트 등록·모델 `qualified` 승격 UI는 후속 범위입니다.
- 대규모 썸네일 가상 스크롤, 화면 위 직접 Box 드래그 편집, CSV/JSON 결과 내보내기는 후속 UI 범위입니다.
- 코드 서명 인증서가 없어 현재 Windows 설치 파일은 서명되지 않았습니다.

통합 요구사항은 [VISIONFORGE_REQUIREMENTS_SPEC.md](VISIONFORGE_REQUIREMENTS_SPEC.md), 상세 기획은 [VISIONFORGE_PRODUCT_PLAN.md](VISIONFORGE_PRODUCT_PLAN.md), 실제 구현 상태는 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), 백엔드 상세는 [HIGH_PERFORMANCE_BACKEND_IMPLEMENTATION.md](HIGH_PERFORMANCE_BACKEND_IMPLEMENTATION.md)를 참고하세요.
