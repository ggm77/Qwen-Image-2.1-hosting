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

type Job = {
  id: string;
  prompt?: string; // task_id 조회로 추가한 작업은 프롬프트를 알 수 없다
  status: string;
  currentStep: number;
  totalSteps: number;
  imageUrl: string | null;
  error: string | null;
  finished: boolean; // 완료·실패·조회 오류로 더 이상 폴링하지 않는 상태
  ratio: number;
  createdAt?: number; // ms
  info?: string; // "512×512 · 20스텝 · seed 42 · int8"
};

type TaskStatus = {
  task_id: string;
  status: string;
  created_at: string;
  current_step: number;
  num_inference_steps: number;
  prompt: string;
  height: number;
  width: number;
  seed: number;
  precision: string;
};

const POLL_INTERVAL_MS = 1500;
const HISTORY_PAGE_SIZE = 50;

// 백엔드 요청 값 범위 (backend/app/schemas/generate/generate.py 와 맞출 것)
const MIN_IMAGE_SIZE = 64;
const MAX_IMAGE_SIZE = 1024;
const IMAGE_SIZE_STEP = 32;
const MAX_STEPS = 100;
const MAX_PROMPT_LENGTH = 2000;
const MAX_PROMPTS_PER_SUBMIT = 50; // 한 번에 큐에 넣을 수 있는 프롬프트 수 (UI 제한)
const MAX_SEED = 2 ** 31 - 1; // 랜덤 시드는 JS 정수 안전 범위 안에서만 뽑는다

// 모든 값이 IMAGE_SIZE_STEP(32)의 배수
const SIZE_PRESETS = [
  { label: "1:1", width: 512, height: 512 },
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

// 숫자 입력은 문자열로 들고 있다가 제출할 때 숫자로 바꾼다. (비워도 0이 남지 않도록)
const digitsOnly = (value: string) => value.replace(/\D/g, "");

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

// 서버의 created_at은 SQLite CURRENT_TIMESTAMP(UTC, "YYYY-MM-DD HH:MM:SS")이다.
const parseServerTime = (value: string): number | undefined => {
  const time = Date.parse(`${value.replace(" ", "T")}Z`);
  return Number.isNaN(time) ? undefined : time;
};

const formatTime = (time: number) =>
  new Date(time).toLocaleString("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

const describeTask = (t: Pick<TaskStatus, "width" | "height" | "num_inference_steps" | "seed" | "precision">) =>
  `${t.width}×${t.height} · ${t.num_inference_steps}스텝 · seed ${t.seed} · ${t.precision}`;

// 서버가 돌려준 작업 하나를 목록 항목으로 바꾼다. 완료된 작업의 이미지는 결과 API를 그대로 가리킨다.
const jobFromTask = (task: TaskStatus): Job => {
  const completed = task.status === "completed";
  const failed = task.status === "failed";
  return {
    id: task.task_id,
    prompt: task.prompt,
    status: task.status,
    currentStep: task.current_step,
    totalSteps: task.num_inference_steps,
    imageUrl: completed ? `/api/v1/tasks/${task.task_id}/result` : null,
    error: failed ? "태스크가 실패했습니다." : null,
    finished: completed || failed,
    ratio: task.width > 0 && task.height > 0 ? task.width / task.height : 1,
    createdAt: parseServerTime(task.created_at),
    info: describeTask(task),
  };
};

const percentOf = (job: Job) =>
  job.totalSteps > 0 ? Math.min(100, Math.round((job.currentStep / job.totalSteps) * 100)) : 0;

type GenerateResponse = { task_id: string; status: string };

// 프롬프트 하나를 큐에 넣고 task_id를 돌려받는다.
const requestGenerate = async (payload: GenerateRequest): Promise<GenerateResponse> => {
  const res = await fetch(`/api/v1/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(await readErrorMessage(res, `이미지 생성 요청 실패: ${res.status}`));
  }
  return res.json();
};

const App: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [height, setHeight] = useState("512");
  const [width, setWidth] = useState("512");
  const [numSteps, setNumSteps] = useState("20");
  const [seed, setSeed] = useState("42");
  const [precision, setPrecision] = useState<Precision>("int8");

  const [lookupTaskId, setLookupTaskId] = useState("");

  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [hasMoreHistory, setHasMoreHistory] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const [copied, setCopied] = useState(false);
  const [theme, setTheme] = useState<Theme>(readStoredTheme);

  const pollTimersRef = useRef<Map<string, number>>(new Map());
  const jobsRef = useRef<Job[]>([]);
  const stageRef = useRef<HTMLElement | null>(null);
  const historyOffsetRef = useRef(0);
  const historyRequestedRef = useRef(false);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  // 언마운트 시 폴링 타이머와 결과 이미지 URL 정리
  useEffect(() => {
    const timers = pollTimersRef.current;
    return () => {
      timers.forEach((timer) => window.clearInterval(timer));
      timers.clear();
      jobsRef.current.forEach((job) => {
        if (job.imageUrl) URL.revokeObjectURL(job.imageUrl);
      });
    };
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

  const updateJob = (id: string, patch: Partial<Job>) => {
    setJobs((prev) => prev.map((job) => (job.id === id ? { ...job, ...patch } : job)));
  };

  const stopPolling = (id: string) => {
    const timer = pollTimersRef.current.get(id);
    if (timer !== undefined) {
      window.clearInterval(timer);
      pollTimersRef.current.delete(id);
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

      setJobs((prev) => {
        // 응답을 기다리는 사이 목록에서 지워졌다면 URL을 바로 해제한다.
        if (!prev.some((job) => job.id === id)) {
          URL.revokeObjectURL(url);
          return prev;
        }
        return prev.map((job) => {
          if (job.id !== id) return job;
          if (job.imageUrl) URL.revokeObjectURL(job.imageUrl);
          return { ...job, imageUrl: url };
        });
      });
    } catch (err) {
      console.error(err);
      updateJob(id, { error: err instanceof Error ? err.message : "알 수 없는 오류입니다." });
    }
  };

  // task_id 하나의 진행 상황을 주기적으로 조회하며 진행률을 갱신한다.
  const startPolling = (id: string) => {
    stopPolling(id);

    const poll = async () => {
      try {
        const res = await fetch(`/api/v1/tasks/${id}`, { method: "GET" });
        if (!res.ok) {
          throw new Error(await readErrorMessage(res, `태스크 상태 요청 실패: ${res.status}`));
        }

        const data: TaskStatus = await res.json();
        // 응답을 기다리는 사이 중단·삭제된 작업이면 결과를 무시한다.
        if (!pollTimersRef.current.has(id)) return;

        updateJob(id, {
          prompt: data.prompt,
          status: data.status,
          currentStep: data.current_step,
          totalSteps: data.num_inference_steps,
          createdAt: parseServerTime(data.created_at),
          info: describeTask(data),
        });

        if (data.status === "completed") {
          stopPolling(id);
          await fetchImage(id);
          updateJob(id, { finished: true });
        } else if (data.status === "failed") {
          stopPolling(id);
          updateJob(id, { finished: true, error: "태스크가 실패했습니다." });
        }
      } catch (err) {
        console.error(err);
        if (!pollTimersRef.current.has(id)) return;
        stopPolling(id);
        updateJob(id, {
          finished: true,
          error: err instanceof Error ? err.message : "알 수 없는 오류입니다.",
        });
      }
    };

    pollTimersRef.current.set(id, window.setInterval(poll, POLL_INTERVAL_MS));
    poll(); // 즉시 1회 조회 후 주기 폴링
  };

  // 서버에 저장된 이전 작업을 최근 순으로 불러와 목록 뒤에 붙인다.
  const loadHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const res = await fetch(`/api/v1/tasks?limit=${HISTORY_PAGE_SIZE}&offset=${historyOffsetRef.current}`);
      if (!res.ok) {
        throw new Error(await readErrorMessage(res, `이전 작업 목록 요청 실패: ${res.status}`));
      }

      const tasks: TaskStatus[] = await res.json();
      historyOffsetRef.current += tasks.length;
      setHasMoreHistory(tasks.length === HISTORY_PAGE_SIZE);

      // 이미 목록에 있는 작업(방금 제출했거나 조회한 것)은 건너뛴다.
      const known = new Set(jobsRef.current.map((job) => job.id));
      const fresh = tasks.filter((task) => !known.has(task.task_id)).map(jobFromTask);

      setJobs((prev) => {
        const ids = new Set(prev.map((job) => job.id));
        return [...prev, ...fresh.filter((job) => !ids.has(job.id))];
      });
      // 서버 재시작 등으로 아직 진행 중인 작업은 이어서 추적한다.
      fresh.filter((job) => !job.finished).forEach((job) => startPolling(job.id));
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "알 수 없는 오류입니다.");
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // 처음 열 때 이전 작업을 한 번 불러온다. (StrictMode의 이중 실행에서도 한 번만 요청)
  useEffect(() => {
    if (historyRequestedRef.current) return;
    historyRequestedRef.current = true;
    void loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 좁은 화면에서는 결과 영역이 폼 아래에 있으므로 시작 시 보이는 위치로 옮긴다.
  const revealStage = () => {
    const el = stageRef.current;
    if (!el) return;
    if (el.getBoundingClientRect().top > window.innerHeight * 0.6) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  // 한 줄에 프롬프트 하나씩. 빈 줄은 건너뛴다.
  const promptLines = prompt
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();

    if (isSubmitting) return;

    if (promptLines.length === 0) {
      setError("prompt를 입력하세요.");
      return;
    }
    if (promptLines.length > MAX_PROMPTS_PER_SUBMIT) {
      setError(`프롬프트는 한 번에 최대 ${MAX_PROMPTS_PER_SUBMIT}개까지 입력할 수 있어요.`);
      return;
    }
    if (promptLines.some((line) => line.length > MAX_PROMPT_LENGTH)) {
      setError(`프롬프트 한 줄은 최대 ${MAX_PROMPT_LENGTH}자까지 입력할 수 있어요.`);
      return;
    }
    if (!width || !height || !numSteps || !seed) {
      setError("가로, 세로, 추론 스텝, 시드를 모두 입력하세요.");
      return;
    }

    const widthValue = Number(width);
    const heightValue = Number(height);
    const stepsValue = Number(numSteps);
    const seedValue = Number(seed);
    const ratio = widthValue > 0 && heightValue > 0 ? widthValue / heightValue : 1;

    setError(null);
    setIsSubmitting(true);

    try {
      // 거의 동시에 큐에 들어가야 백엔드가 한 배치로 묶어 인코딩할 수 있다.
      const results = await Promise.allSettled(
        promptLines.map((line) =>
          requestGenerate({
            prompt: line,
            height: heightValue,
            width: widthValue,
            num_inference_steps: stepsValue,
            seed: seedValue,
            precision,
          }),
        ),
      );

      const created: Job[] = [];
      const failedLines: string[] = [];
      let firstError: string | null = null;

      results.forEach((result, i) => {
        if (result.status === "fulfilled") {
          created.push({
            id: result.value.task_id,
            prompt: promptLines[i],
            status: result.value.status,
            currentStep: 0,
            totalSteps: stepsValue,
            imageUrl: null,
            error: null,
            finished: false,
            ratio,
            createdAt: Date.now(),
            info: `${widthValue}×${heightValue} · ${stepsValue}스텝 · seed ${seedValue} · ${precision}`,
          });
        } else {
          console.error(result.reason);
          failedLines.push(promptLines[i]);
          firstError ??= result.reason instanceof Error ? result.reason.message : "알 수 없는 오류입니다.";
        }
      });

      if (created.length > 0) {
        setJobs((prev) => [...created, ...prev]);
        setSelectedId(created[0].id);
        created.forEach((job) => startPolling(job.id));
        revealStage();
      }

      if (failedLines.length > 0) {
        // 실패한 줄만 남겨 두면 그대로 다시 제출할 수 있다.
        setPrompt(failedLines.join("\n"));
        setError(
          created.length > 0
            ? `${promptLines.length}개 중 ${failedLines.length}개 요청 실패: ${firstError}`
            : firstError,
        );
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // 입력한 task_id로 진행 상황을 조회하고, 완료 상태면 이어서 폴링한다.
  const handleLookup = (e: React.FormEvent) => {
    e.preventDefault();

    const id = lookupTaskId.trim();
    if (!id) {
      setError("조회할 task_id를 입력하세요.");
      return;
    }

    setError(null);

    // 목록에 이미 있는 작업이면 새로 만들지 않고 선택만 한다.
    if (jobsRef.current.some((job) => job.id === id)) {
      setSelectedId(id);
      revealStage();
      return;
    }

    const job: Job = {
      id,
      status: "",
      currentStep: 0,
      totalSteps: 0,
      imageUrl: null,
      error: null,
      finished: false,
      ratio: 1,
    };
    setJobs((prev) => [job, ...prev]);
    setSelectedId(id);
    startPolling(id);
    revealStage();
  };

  const removeJobs = (ids: Set<string>) => {
    ids.forEach(stopPolling);
    const remaining = jobs.filter((job) => !ids.has(job.id));
    jobs.forEach((job) => {
      if (ids.has(job.id) && job.imageUrl) URL.revokeObjectURL(job.imageUrl);
    });
    setJobs(remaining);
    if (selectedId && ids.has(selectedId)) {
      setSelectedId(remaining[0]?.id ?? null);
    }
  };

  const handlePromptKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      e.currentTarget.form?.requestSubmit();
    }
  };

  const selected = jobs.find((job) => job.id === selectedId) ?? null;

  const handleCopyTaskId = async () => {
    if (!selected) return;
    try {
      await navigator.clipboard.writeText(selected.id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 클립보드 접근이 막힌 환경에서는 조용히 무시
    }
  };

  const closeError = () => {
    setError(null);
    if (selected?.error) updateJob(selected.id, { error: null });
  };

  const progressPercent = selected ? percentOf(selected) : 0;
  const selectedStatus = selected?.status ?? "";
  const statusLabel = STATUS_LABELS[selectedStatus] ?? selectedStatus;
  const isFailed = selectedStatus === "failed";
  const isCompleted = selectedStatus === "completed";
  const isBusy = selected !== null && !selected.finished;
  const shownError = error ?? selected?.error ?? null;
  const hasFinishedJobs = jobs.some((job) => job.finished);
  const sliderSteps = Math.min(Math.max(Number(numSteps) || 1, 1), MAX_STEPS);
  const stepsFill = `${((sliderSteps - 1) / (MAX_STEPS - 1)) * 100}%`;
  const previewRatio = selected?.ratio ?? 1;

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
                  {promptLines.length > 1 ? `${promptLines.length}개 · ` : ""}
                  {prompt.length}자
                </span>
              </div>
              <textarea
                id="prompt"
                className="textarea"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handlePromptKeyDown}
                placeholder={"한 줄에 프롬프트 하나씩 입력하면 여러 장을 한 번에 생성해요.\n예) 노을 지는 바닷가의 작은 등대, 수채화 스타일\n우주복을 입은 고양이"}
              />
              <span className="hint">줄바꿈은 Enter, 생성은 ⌘/Ctrl + Enter</span>
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
                      aria-pressed={width === String(p.width) && height === String(p.height)}
                      onClick={() => {
                        setWidth(String(p.width));
                        setHeight(String(p.height));
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
                    type="text"
                    inputMode="numeric"
                    autoComplete="off"
                    aria-label="가로(px)"
                    value={width}
                    onChange={(e) => setWidth(digitsOnly(e.target.value))}
                  />
                  <span className="suffix">W</span>
                </div>
                <div className="with-suffix">
                  <input
                    className="input"
                    type="text"
                    inputMode="numeric"
                    autoComplete="off"
                    aria-label="세로(px)"
                    value={height}
                    onChange={(e) => setHeight(digitsOnly(e.target.value))}
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
                  type="text"
                  inputMode="numeric"
                  autoComplete="off"
                  aria-label="추론 스텝 수"
                  value={numSteps}
                  onChange={(e) => setNumSteps(digitsOnly(e.target.value))}
                  style={{ width: 72, padding: "5px 8px", textAlign: "center" }}
                />
              </div>
              <input
                id="steps"
                className="range"
                type="range"
                min={1}
                max={MAX_STEPS}
                value={sliderSteps}
                onChange={(e) => setNumSteps(e.target.value)}
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
                  type="text"
                  inputMode="numeric"
                  autoComplete="off"
                  value={seed}
                  onChange={(e) => setSeed(digitsOnly(e.target.value))}
                />
                <button
                  type="button"
                  className="btn btn-icon"
                  onClick={() => setSeed(String(Math.floor(Math.random() * MAX_SEED)))}
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

            <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  요청 중...
                </>
              ) : (
                <>
                  <SparklesIcon size={18} />
                  {promptLines.length > 1 ? `이미지 ${promptLines.length}장 생성` : "이미지 생성"}
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
          {shownError && (
            <div className="alert" role="alert">
              <AlertIcon />
              <span className="alert-text">{shownError}</span>
              <button type="button" className="alert-close" onClick={closeError} aria-label="오류 닫기">
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
              {selected?.imageUrl ? (
                <img
                  key={selected.id}
                  className="preview-img"
                  src={selected.imageUrl}
                  alt="생성된 이미지"
                  onError={() =>
                    updateJob(selected.id, { imageUrl: null, error: "결과 이미지를 불러올 수 없어요." })
                  }
                  onLoad={(e) => {
                    const { naturalWidth, naturalHeight } = e.currentTarget;
                    if (naturalWidth > 0 && naturalHeight > 0) {
                      updateJob(selected.id, { ratio: naturalWidth / naturalHeight });
                    }
                  }}
                />
              ) : isBusy ? (
                <div className="preview-busy">
                  <span className="busy-percent">{progressPercent}%</span>
                  <span className="busy-label">
                    {selected.status === "queued" || selected.totalSteps === 0
                      ? "대기 중..."
                      : `${selected.currentStep} / ${selected.totalSteps} 스텝`}
                  </span>
                </div>
              ) : (
                <div className="preview-empty">
                  <div className="empty-icon">
                    <ImageIcon />
                  </div>
                  <p className="empty-title">
                    {selected ? "표시할 이미지가 없어요" : "아직 생성된 이미지가 없어요"}
                  </p>
                  <p className="empty-desc">
                    프롬프트를 입력하고 생성 버튼을 누르거나, task_id로 이전 작업을 조회해 보세요.
                  </p>
                </div>
              )}
            </div>

            {selected?.imageUrl && (
              <div className="preview-actions">
                <span className="preview-meta">{selected.id}</span>
                <a className="btn btn-sm" href={selected.imageUrl} download={`${selected.id}.png`}>
                  <DownloadIcon />
                  PNG 다운로드
                </a>
              </div>
            )}
          </div>

          <div className="card task-card">
            <h2 className="section-title">작업 상태</h2>
            {selected ? (
              <>
                {selected.prompt && <p className="task-prompt">{selected.prompt}</p>}
                {selected.info && <p className="task-info">{selected.info}</p>}
                <div className="task-head">
                  <div className="task-id">
                    <span className="task-id-text" title={selected.id}>
                      {selected.id}
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
                  {selectedStatus.length > 0 && (
                    <span className={`badge badge-${selectedStatus}`}>{statusLabel}</span>
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
                  <span>
                    {selected.totalSteps > 0 ? `${selected.currentStep} / ${selected.totalSteps} 스텝` : "대기 중..."}
                  </span>
                  <span>{progressPercent}%</span>
                </div>
              </>
            ) : (
              <p className="task-empty">태스크가 조회되지 않았어요.</p>
            )}
          </div>

          {jobs.length > 0 && (
            <div className="card task-card">
              <div className="list-head">
                <h2 className="section-title">작업 목록 ({jobs.length})</h2>
                {hasFinishedJobs && (
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => removeJobs(new Set(jobs.filter((job) => job.finished).map((job) => job.id)))}
                  >
                    완료 항목 지우기
                  </button>
                )}
              </div>
              <ul className="job-list">
                {jobs.map((job) => {
                  const percent = percentOf(job);
                  const label = STATUS_LABELS[job.status] ?? job.status;
                  return (
                    <li key={job.id} className={`job${job.id === selectedId ? " is-selected" : ""}`}>
                      <button
                        type="button"
                        className="job-main"
                        onClick={() => setSelectedId(job.id)}
                        aria-current={job.id === selectedId}
                      >
                        <span className="job-thumb">
                          {job.imageUrl ? (
                            <img
                              src={job.imageUrl}
                              alt=""
                              loading="lazy"
                              onError={() => updateJob(job.id, { imageUrl: null })}
                            />
                          ) : job.finished ? (
                            <ImageIcon size={18} />
                          ) : (
                            <span className="job-spin" aria-hidden="true" />
                          )}
                        </span>
                        <span className="job-body">
                          <span className={`job-title${job.prompt ? "" : " is-mono"}`}>{job.prompt ?? job.id}</span>
                          <span className="job-sub">
                            {job.status && <span className={`badge badge-${job.status}`}>{label}</span>}
                            {job.finished && job.status === "completed" ? null : <span>{percent}%</span>}
                            {job.createdAt !== undefined && <span>{formatTime(job.createdAt)}</span>}
                          </span>
                        </span>
                      </button>
                      <button
                        type="button"
                        className="job-remove"
                        onClick={() => removeJobs(new Set([job.id]))}
                        aria-label="목록에서 제거"
                        title="목록에서 제거"
                      >
                        <CloseIcon size={14} />
                      </button>
                    </li>
                  );
                })}
              </ul>
              {hasMoreHistory && (
                <button type="button" className="btn btn-sm load-more" onClick={loadHistory} disabled={isLoadingHistory}>
                  {isLoadingHistory ? "불러오는 중..." : "이전 작업 더 보기"}
                </button>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default App;
