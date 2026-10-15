// src/ImageAnnotator.tsx
// 첨부한 이미지 위에 빨간 펜으로 표시하는 편집기. 모델이 이미지에 그려 넣은 표시를 편집 지시로 읽으므로,
// 적용하면 표시가 합성된 이미지를 돌려준다.
import React, { useCallback, useEffect, useRef, useState } from "react";

const PEN_COLOR = "#ff0000";
// 서버가 입력 이미지를 약 768² 면적으로 줄여 쓰므로 그보다 충분히 큰 값으로 작업 크기를 제한한다.
const MAX_EDIT_SIDE = 1536;
const MIN_BRUSH = 2;
const MAX_BRUSH = 40;
const DEFAULT_BRUSH = 8; // 화면에서 보이는 굵기(px)

type Point = { x: number; y: number };
type Stroke = { width: number; points: Point[] }; // 모두 캔버스 좌표계

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

export const PenIcon = ({ size = 16 }: { size?: number }) => (
  <svg {...iconProps(size)}>
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
  </svg>
);

const UndoIcon = () => (
  <svg {...iconProps(15)}>
    <path d="M9 14L4 9l5-5" />
    <path d="M4 9h10a6 6 0 0 1 0 12h-3" />
  </svg>
);

const TrashIcon = () => (
  <svg {...iconProps(15)}>
    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />
  </svg>
);

type Props = {
  src: string; // 지금 첨부된 이미지 (data URL)
  originalSrc: string; // 표시를 하기 전의 원본. 처음 편집하면 src와 같다.
  title: string;
  onSave: (dataUrl: string) => void;
  onCancel: () => void;
};

const ImageAnnotator: React.FC<Props> = ({ src, originalSrc, title, onSave, onCancel }) => {
  const [base, setBase] = useState(src);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [brush, setBrush] = useState(DEFAULT_BRUSH);
  const [loaded, setLoaded] = useState(0); // 이미지를 (다시) 불러온 횟수. 그릴 때마다 다시 그리는 데 쓴다.
  const [loadFailed, setLoadFailed] = useState(false);

  const dialogRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const currentRef = useRef<Stroke | null>(null); // 지금 긋고 있는 선

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !image || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = PEN_COLOR;
    ctx.fillStyle = PEN_COLOR;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const all = currentRef.current ? [...strokes, currentRef.current] : strokes;
    for (const stroke of all) {
      const [first, ...rest] = stroke.points;
      if (rest.length === 0) {
        ctx.beginPath(); // 클릭만 했으면 점 하나
        ctx.arc(first.x, first.y, stroke.width / 2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      ctx.lineWidth = stroke.width;
      ctx.beginPath();
      ctx.moveTo(first.x, first.y);
      rest.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    }
  }, [strokes]);

  // 바탕 이미지를 불러와 캔버스 크기를 맞춘다. 선이 바뀔 때마다 다시 불러오지 않도록 redraw는 ref로 부른다.
  useEffect(() => {
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      const canvas = canvasRef.current;
      if (cancelled || !canvas) return;
      const scale = Math.min(1, MAX_EDIT_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      imageRef.current = image;
      setLoadFailed(false);
      setLoaded((n) => n + 1);
    };
    image.onerror = () => {
      if (!cancelled) setLoadFailed(true);
    };
    image.src = base;
    return () => {
      cancelled = true;
    };
  }, [base]);

  useEffect(() => {
    redraw();
  }, [redraw, loaded]);

  // 열려 있는 동안 배경 스크롤을 막고, 키보드 입력(Esc 닫기, ⌘/Ctrl+Z 되돌리기)을 받는다.
  useEffect(() => {
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialogRef.current?.focus();
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, []);

  const undo = () => setStrokes((prev) => prev.slice(0, -1));

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      undo();
    }
  };

  const toCanvasPoint = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) * canvas.width) / rect.width,
      y: ((e.clientY - rect.top) * canvas.height) / rect.height,
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const canvas = e.currentTarget;
    canvas.setPointerCapture(e.pointerId);
    // 화면에 보이는 굵기가 이미지를 줄여 보여 주는 정도와 상관없이 같게 보이도록 캔버스 좌표계로 환산한다.
    const width = (brush * canvas.width) / canvas.getBoundingClientRect().width;
    currentRef.current = { width, points: [toCanvasPoint(e)] };
    redraw();
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const stroke = currentRef.current;
    if (!stroke) return;
    stroke.points.push(toCanvasPoint(e));
    redraw();
  };

  const endStroke = () => {
    const stroke = currentRef.current;
    currentRef.current = null;
    if (stroke) setStrokes((prev) => [...prev, stroke]);
  };

  const canReset = base !== originalSrc;

  const handleSave = () => {
    const canvas = canvasRef.current;
    if (strokes.length > 0 && canvas) {
      onSave(canvas.toDataURL("image/png"));
    } else if (canReset) {
      onSave(originalSrc); // 원본으로 되돌리기만 한 경우
    } else {
      onCancel(); // 바뀐 것이 없다
    }
  };

  const brushFill = `${((brush - MIN_BRUSH) / (MAX_BRUSH - MIN_BRUSH)) * 100}%`;

  return (
    <div className="editor-backdrop">
      <div
        className="editor"
        role="dialog"
        aria-modal="true"
        aria-label={`${title} 편집`}
        tabIndex={-1}
        ref={dialogRef}
        onKeyDown={handleKeyDown}
      >
        <div className="editor-bar">
          <span className="editor-title">
            <PenIcon />
            빨간 펜으로 표시
          </span>
          <label className="editor-size">
            <span className="hint">굵기</span>
            <input
              className="range"
              type="range"
              min={MIN_BRUSH}
              max={MAX_BRUSH}
              value={brush}
              onChange={(e) => setBrush(Number(e.target.value))}
              style={{ "--fill": brushFill } as React.CSSProperties}
              aria-label="펜 굵기"
            />
            <span className="hint editor-size-value">{brush}</span>
          </label>
          <div className="editor-tools">
            <button type="button" className="btn btn-sm" onClick={undo} disabled={strokes.length === 0}>
              <UndoIcon />
              되돌리기
            </button>
            <button type="button" className="btn btn-sm" onClick={() => setStrokes([])} disabled={strokes.length === 0}>
              <TrashIcon />
              모두 지우기
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                setBase(originalSrc);
                setStrokes([]);
              }}
              disabled={!canReset}
              title="이전에 적용한 표시까지 모두 없애고 처음 업로드한 이미지로 돌아가요"
            >
              원본으로
            </button>
          </div>
        </div>

        <div className="editor-stage">
          {loadFailed ? (
            <p className="editor-error">이미지를 불러오지 못했어요.</p>
          ) : (
            <canvas
              ref={canvasRef}
              className="editor-canvas"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={endStroke}
              onPointerCancel={endStroke}
            />
          )}
        </div>

        <div className="editor-foot">
          <span className="hint">고칠 부분을 빨간 펜으로 표시하고, 프롬프트에 “빨간 펜으로 표시한 부분을 …”처럼 적어 주세요.</span>
          <div className="editor-actions">
            <button type="button" className="btn" onClick={onCancel}>
              취소
            </button>
            <button type="button" className="btn btn-primary" onClick={handleSave} disabled={loadFailed}>
              적용
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageAnnotator;
