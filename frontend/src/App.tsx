// src/App.tsx
import React, { useState, useEffect, useRef } from "react";
import "./App.css";

type Precision = "bf16" | "int8";
type Theme = "system" | "light" | "dark";

type GenerateRequest = {
  prompt: string;
  height: number;
  width: number;
  num_inference_steps: number;
  seed: number;
  precision: Precision;
};

type TaskStatus = {
  task_id: string;
  status: string;
  created_at: string;
  current_step: number;
  num_inference_steps: number;
};

const POLL_INTERVAL_MS = 1500;

// 백엔드 요청 값 범위 (backend/app/schemas/generate/generate.py 와 맞출 것)
const MIN_IMAGE_SIZE = 64;
const MAX_IMAGE_SIZE = 1024;
const IMAGE_SIZE_STEP = 32;
const MAX_STEPS = 100;
const MAX_PROMPT_LENGTH = 2000;
const MAX_SEED = 2 ** 31 - 1; // 랜덤 시드는 JS 정수 안전 범위 안에서만 뽑는다

// 모든 값이 IMAGE_SIZE_STEP(32)의 배수
const SIZE_PRESETS = [
  { label: "1:1", width: 768, height: 768 },
  { label: "4:3", width: 1024, height: 768 },
  { label: "3:4", width: 768, height: 1024 },
  { label: "16:9", width: 1024, height: 576 },
  { label: "9:16", width: 576, height: 1024 },
] as const;

const PRECISIONS: { value: Precision; title: string; desc: string }[] = [
  { value: "bf16", title: "bf16", desc: "원본 · 메모리 많이 사용" },
  { value: "int8", title: "int8", desc: "양자화 · 더 빠름" },
];

const STATUS_LABELS: Record<string, string> = {
  queued: "대기 중",
  processing: "생성 중",
  completed: "완료",
  failed: "실패",
};

const THEME_STORAGE_KEY = "theme";

type ValidationDetail = { loc?: (string | number)[]; msg: string };

// 백엔드가 돌려준 오류 메시지(detail)를 읽는다. 없으면 fallback을 쓴다.
const readErrorMessage = async (res: Response, fallback: string): Promise<string> => {
  try {
    const body = await res.json();
    const detail: unknown = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return (detail as ValidationDetail[])
        .map((d) => `${d.loc?.[d.loc.length - 1] ?? "입력값"}: ${d.msg}`)
        .join(", ");
    }
  } catch {
    // JSON이 아닌 응답이면 fallback 사용
  }
  return fallback;
};

const readStoredTheme = (): Theme => {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // 저장소를 쓸 수 없으면 시스템 설정을 따른다
  }
  return "system";
};

/* ---------- Icons ---------- */
type IconProps = { size?: number };

const iconProps = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
});

const SparklesIcon = ({ size = 20 }: IconProps) => (
  <svg {...iconProps(size)}>
    <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" />
    <path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15z" />
  </svg>
);

const ImageIcon = ({ size = 24 }: IconProps) => (
  <svg {...iconProps(size)}>
    <rect x="3" y="3" width="18" height="18" rx="3" />
    <circle cx="9" cy="9" r="1.8" />
    <path d="M21 16l-5-5-9 9" />
  </svg>
);

const DiceIcon = ({ size = 18 }: IconProps) => (
  <svg {...iconProps(size)}>
    <rect x="3" y="3" width="18" height="18" rx="4" />
    <circle cx="8.5" cy="8.5" r=".8" fill="currentColor" />
    <circle cx="15.5" cy="8.5" r=".8" fill="currentColor" />
    <circle cx="12" cy="12" r=".8" fill="currentColor" />
    <circle cx="8.5" cy="15.5" r=".8" fill="currentColor" />
    <circle cx="15.5" cy="15.5" r=".8" fill="currentColor" />
  </svg>
);

const SearchIcon = ({ size = 18 }: IconProps) => (
  <svg {...iconProps(size)}>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.3-4.3" />
  </svg>
);

const CopyIcon = ({ size = 14 }: IconProps) => (
  <svg {...iconProps(size)}>
    <rect x="9" y="9" width="12" height="12" rx="2.5" />
    <path d="M5 15V6a2 2 0 0 1 2-2h9" />
  </svg>
);

const CheckIcon = ({ size = 14 }: IconProps) => (
  <svg {...iconProps(size)}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </svg>
);

const DownloadIcon = ({ size = 16 }: IconProps) => (
  <svg {...iconProps(size)}>
    <path d="M12 3v12" />
    <path d="M7 11l5 5 5-5" />
    <path d="M5 21h14" />
  </svg>
);

const AlertIcon = ({ size = 18 }: IconProps) => (
  <svg {...iconProps(size)}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v5" />
    <circle cx="12" cy="16.5" r=".6" fill="currentColor" />
  </svg>
);

const CloseIcon = ({ size = 16 }: IconProps) => (
  <svg {...iconProps(size)}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

const SunIcon = ({ size = 18 }: IconProps) => (
  <svg {...iconProps(size)}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);

const MoonIcon = ({ size = 18 }: IconProps) => (
  <svg {...iconProps(size)}>
    <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5a8.5 8.5 0 1 0 10.7 10.7z" />
  </svg>
);

const App: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [height, setHeight] = useState(768);
  const [width, setWidth] = useState(768);
  const [numSteps, setNumSteps] = useState(20);
  const [seed, setSeed] = useState(42);
  const [precision, setPrecision] = useState<Precision>("bf16");

  const [taskId, setTaskId] = useState("");
  const [lookupTaskId, setLookupTaskId] = useState("");
  const [taskStatus, setTaskStatus] = useState("");
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [previewRatio, setPreviewRatio] = useState(1);
  const [copied, setCopied] = useState(false);
  const [theme, setTheme] = useState<Theme>(readStoredTheme);

  const pollTimerRef = useRef<number | null>(null);
  const stageRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  // 언마운트 시 폴링 타이머 정리
  useEffect(() => {
    return () => stopPolling();
  }, []);

  // 테마 적용: system이면 속성을 제거해 prefers-color-scheme을 따른다
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", theme);
    }
  }, [theme]);

  const isDarkNow = () =>
    theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  const toggleTheme = () => {
    const next: Theme = isDarkNow() ? "light" : "dark";
    setTheme(next);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // 저장 실패는 무시 (이번 세션에서만 적용)
    }
  };

  const stopPolling = () => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const fetchImage = async (id: string) => {
    try {
      const res = await fetch(`/api/v1/tasks/${id}/result`, { method: "GET" });
      if (!res.ok) {
        throw new Error(await readErrorMessage(res, `이미지 생성 결과 요청 실패: ${res.status}`));
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      setImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "알 수 없는 오류입니다.");
    }
  };

  // task_id 하나의 진행 상황을 주기적으로 조회하며 진행률을 갱신한다.
  const startPolling = (id: string) => {
    stopPolling();

    const poll = async () => {
      try {
        const res = await fetch(`/api/v1/tasks/${id}`, { method: "GET" });
        if (!res.ok) {
          throw new Error(await readErrorMessage(res, `태스크 상태 요청 실패: ${res.status}`));
        }

        const data: TaskStatus = await res.json();
        setTaskStatus(data.status);
        setCurrentStep(data.current_step);
        setTotalSteps(data.num_inference_steps);

        if (data.status === "completed") {
          stopPolling();
          setIsGenerating(false);
          await fetchImage(id);
        } else if (data.status === "failed") {
          stopPolling();
          setIsGenerating(false);
          setError("태스크가 실패했습니다.");
        }
      } catch (err) {
        console.error(err);
        stopPolling();
        setIsGenerating(false);
        setError(err instanceof Error ? err.message : "알 수 없는 오류입니다.");
      }
    };

    poll(); // 즉시 1회 조회 후 주기 폴링 시작
    pollTimerRef.current = window.setInterval(poll, POLL_INTERVAL_MS);
  };

  // 좁은 화면에서는 결과 영역이 폼 아래에 있으므로 시작 시 보이는 위치로 옮긴다.
  const revealStage = () => {
    const el = stageRef.current;
    if (!el) return;
    if (el.getBoundingClientRect().top > window.innerHeight * 0.6) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();

    if (isGenerating) return;

    if (!prompt.trim()) {
      setError("prompt를 입력하세요.");
      return;
    }

    setError(null);
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setTaskStatus("");
    setCurrentStep(0);
    setTotalSteps(numSteps);
    setPreviewRatio(width / height);

    try {
      const payload: GenerateRequest = {
        prompt,
        height,
        width,
        num_inference_steps: numSteps,
        seed,
        precision,
      };

      const res = await fetch(`/api/v1/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(await readErrorMessage(res, `이미지 생성 요청 실패: ${res.status}`));
      }

      const data = await res.json();
      setTaskId(data.task_id);
      setTaskStatus(data.status);
      setIsGenerating(true);

      startPolling(data.task_id);
      revealStage();
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "알 수 없는 오류입니다.");
      setIsGenerating(false);
    }
  };

  // 입력한 task_id로 진행 상황을 조회하고, 완료 상태면 이어서 폴링한다.
  const handleLookup = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!lookupTaskId.trim()) {
      setError("조회할 task_id를 입력하세요.");
      return;
    }

    // 이전 작업의 상태·진행률·이미지가 남아 새 조회 결과와 섞여 보이지 않도록 초기화
    setError(null);
    setTaskStatus("");
    setCurrentStep(0);
    setTotalSteps(0);
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setTaskId(lookupTaskId.trim());
    setIsGenerating(true);
    startPolling(lookupTaskId.trim());
    revealStage();
  };

  const handlePromptKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      e.currentTarget.form?.requestSubmit();
    }
  };

  const handleCopyTaskId = async () => {
    try {
      await navigator.clipboard.writeText(taskId);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 클립보드 접근이 막힌 환경에서는 조용히 무시
    }
  };

  const progressPercent =
    totalSteps > 0 ? Math.min(100, Math.round((currentStep / totalSteps) * 100)) : 0;

  const statusLabel = STATUS_LABELS[taskStatus] ?? taskStatus;
  const isFailed = taskStatus === "failed";
  const isCompleted = taskStatus === "completed";
  const isBusy = isGenerating && !imageUrl;
  const stepsFill = `${((numSteps - 1) / (MAX_STEPS - 1)) * 100}%`;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <SparklesIcon />
          </div>
          <div>
            <h1 className="brand-name">Qwen Image Studio</h1>
            <p className="brand-sub">텍스트로 이미지를 만들어 보세요</p>
          </div>
        </div>
        <button
          type="button"
          className="btn theme-toggle"
          onClick={toggleTheme}
          aria-label="테마 전환"
          title="테마 전환"
        >
          {isDarkNow() ? <SunIcon /> : <MoonIcon />}
        </button>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <form onSubmit={handleGenerate} className="card card-pad form">
            <div className="field">
              <div className="field-head">
                <label className="label" htmlFor="prompt">
                  프롬프트
                </label>
                <span className="hint">
                  {prompt.length} / {MAX_PROMPT_LENGTH}
                </span>
              </div>
              <textarea
                id="prompt"
                className="textarea"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handlePromptKeyDown}
                maxLength={MAX_PROMPT_LENGTH}
                placeholder="만들고 싶은 이미지를 자세히 설명해 주세요. 예) 노을 지는 바닷가의 작은 등대, 수채화 스타일"
              />
            </div>

            <div className="field">
              <span className="label">비율</span>
              <div className="ratios" role="group" aria-label="비율 프리셋">
                {SIZE_PRESETS.map((p) => {
                  const longSide = 20;
                  const w = p.width >= p.height ? longSide : Math.round((longSide * p.width) / p.height);
                  const h = p.height >= p.width ? longSide : Math.round((longSide * p.height) / p.width);
                  return (
                    <button
                      key={p.label}
                      type="button"
                      className="ratio"
                      aria-pressed={width === p.width && height === p.height}
                      onClick={() => {
                        setWidth(p.width);
                        setHeight(p.height);
                      }}
                    >
                      <span className="ratio-box">
                        <span className="ratio-shape" style={{ width: w, height: h }} />
                      </span>
                      {p.label}
                    </button>
                  );
                })}
              </div>
              <div className="row-2">
                <div className="with-suffix">
                  <input
                    className="input"
                    type="number"
                    aria-label="가로(px)"
                    min={MIN_IMAGE_SIZE}
                    max={MAX_IMAGE_SIZE}
                    step={IMAGE_SIZE_STEP}
                    value={width}
                    onChange={(e) => setWidth(Number(e.target.value))}
                  />
                  <span className="suffix">W</span>
                </div>
                <div className="with-suffix">
                  <input
                    className="input"
                    type="number"
                    aria-label="세로(px)"
                    min={MIN_IMAGE_SIZE}
                    max={MAX_IMAGE_SIZE}
                    step={IMAGE_SIZE_STEP}
                    value={height}
                    onChange={(e) => setHeight(Number(e.target.value))}
                  />
                  <span className="suffix">H</span>
                </div>
              </div>
              <span className="hint">
                {MIN_IMAGE_SIZE}~{MAX_IMAGE_SIZE}px, {IMAGE_SIZE_STEP}의 배수
              </span>
            </div>

            <div className="field">
              <div className="field-head">
                <label className="label" htmlFor="steps">
                  추론 스텝
                </label>
                <input
                  className="input steps-value"
                  type="number"
                  aria-label="추론 스텝 수"
                  min={1}
                  max={MAX_STEPS}
                  value={numSteps}
                  onChange={(e) => setNumSteps(Number(e.target.value))}
                  style={{ width: 72, padding: "5px 8px", textAlign: "center" }}
                />
              </div>
              <input
                id="steps"
                className="range"
                type="range"
                min={1}
                max={MAX_STEPS}
                value={Math.min(Math.max(numSteps, 1), MAX_STEPS)}
                onChange={(e) => setNumSteps(Number(e.target.value))}
                style={{ "--fill": stepsFill } as React.CSSProperties}
              />
            </div>

            <div className="field">
              <label className="label" htmlFor="seed">
                시드
              </label>
              <div className="input-group">
                <input
                  id="seed"
                  className="input"
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
                <button
                  type="button"
                  className="btn btn-icon"
                  onClick={() => setSeed(Math.floor(Math.random() * MAX_SEED))}
                  aria-label="랜덤 시드"
                  title="랜덤 시드"
                >
                  <DiceIcon />
                </button>
              </div>
            </div>

            <div className="field">
              <span className="label">정밀도</span>
              <div className="segmented" role="group" aria-label="정밀도">
                {PRECISIONS.map((p) => (
                  <button
                    key={p.value}
                    type="button"
                    className="segment"
                    aria-pressed={precision === p.value}
                    onClick={() => setPrecision(p.value)}
                  >
                    <span className="segment-title">{p.title}</span>
                    <span className="segment-desc">{p.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            <button type="submit" className="btn btn-primary" disabled={isGenerating}>
              {isGenerating ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  생성 중...
                </>
              ) : (
                <>
                  <SparklesIcon size={18} />
                  이미지 생성
                </>
              )}
            </button>
          </form>

          <form onSubmit={handleLookup} className="card card-pad">
            <h2 className="section-title">작업 조회</h2>
            <div className="lookup">
              <input
                className="input input-mono"
                type="text"
                placeholder="task_id 입력"
                aria-label="조회할 task_id"
                value={lookupTaskId}
                onChange={(e) => setLookupTaskId(e.target.value)}
              />
              <button type="submit" className="btn btn-icon" aria-label="진행상황 조회" title="진행상황 조회">
                <SearchIcon />
              </button>
            </div>
          </form>
        </aside>

        <section className="stage" ref={stageRef} aria-live="polite">
          {error && (
            <div className="alert" role="alert">
              <AlertIcon />
              <span className="alert-text">{error}</span>
              <button type="button" className="alert-close" onClick={() => setError(null)} aria-label="오류 닫기">
                <CloseIcon />
              </button>
            </div>
          )}

          <div className="card preview-card">
            <div
              className="preview-frame"
              style={{
                aspectRatio: String(previewRatio),
                width: `min(100%, calc(72vh * ${previewRatio}))`,
              }}
            >
              {imageUrl ? (
                <img
                  className="preview-img"
                  src={imageUrl}
                  alt="생성된 이미지"
                  onLoad={(e) => {
                    const { naturalWidth, naturalHeight } = e.currentTarget;
                    if (naturalWidth > 0 && naturalHeight > 0) {
                      setPreviewRatio(naturalWidth / naturalHeight);
                    }
                  }}
                />
              ) : isBusy ? (
                <div className="preview-busy">
                  <span className="busy-percent">{progressPercent}%</span>
                  <span className="busy-label">
                    {totalSteps > 0 ? `${currentStep} / ${totalSteps} 스텝` : "대기 중..."}
                  </span>
                </div>
              ) : (
                <div className="preview-empty">
                  <div className="empty-icon">
                    <ImageIcon />
                  </div>
                  <p className="empty-title">아직 생성된 이미지가 없어요</p>
                  <p className="empty-desc">
                    프롬프트를 입력하고 생성 버튼을 누르거나, task_id로 이전 작업을 조회해 보세요.
                  </p>
                </div>
              )}
            </div>

            {imageUrl && (
              <div className="preview-actions">
                <span className="preview-meta">{taskId}</span>
                <a className="btn btn-sm" href={imageUrl} download={`${taskId}.png`}>
                  <DownloadIcon />
                  PNG 다운로드
                </a>
              </div>
            )}
          </div>

          <div className="card task-card">
            <h2 className="section-title">작업 상태</h2>
            {taskId.length > 0 ? (
              <>
                <div className="task-head">
                  <div className="task-id">
                    <span className="task-id-text" title={taskId}>
                      {taskId}
                    </span>
                    <button
                      type="button"
                      className="copy-btn"
                      onClick={handleCopyTaskId}
                      aria-label="task_id 복사"
                      title={copied ? "복사됨" : "task_id 복사"}
                    >
                      {copied ? <CheckIcon /> : <CopyIcon />}
                    </button>
                  </div>
                  {taskStatus.length > 0 && (
                    <span className={`badge badge-${taskStatus}`}>{statusLabel}</span>
                  )}
                </div>
                <div
                  className="progress"
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={progressPercent}
                >
                  <div
                    className={`progress-bar${isFailed ? " is-failed" : isCompleted ? " is-done" : ""}`}
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
                <div className="progress-meta">
                  <span>{totalSteps > 0 ? `${currentStep} / ${totalSteps} 스텝` : "대기 중..."}</span>
                  <span>{progressPercent}%</span>
                </div>
              </>
            ) : (
              <p className="task-empty">태스크가 조회되지 않았어요.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
