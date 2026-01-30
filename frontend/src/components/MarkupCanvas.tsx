"use client";

import { useRef, useEffect, useState, useCallback } from "react";

interface MarkupCanvasProps {
  imageUrl: string;
  width: number;
  height: number;
  brushColor?: string;
  brushSize?: number;
  onMarkupChange?: (hasMarkup: boolean) => void;
  onExport?: (blob: Blob) => void;
}

interface Point {
  x: number;
  y: number;
}

interface Stroke {
  points: Point[];
  color: string;
  size: number;
}

export default function MarkupCanvas({
  imageUrl,
  width,
  height,
  brushColor = "rgba(255, 200, 0, 0.3)",
  brushSize = 30,
  onMarkupChange,
  onExport,
}: MarkupCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [currentStroke, setCurrentStroke] = useState<Point[]>([]);
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imageRef.current = img;
      redraw();
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Redraw canvas
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !imageRef.current) return;

    // Clear and draw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);

    // Draw all strokes
    [...strokes, { points: currentStroke, color: brushColor, size: brushSize }].forEach(
      (stroke) => {
        if (stroke.points.length < 2) return;

        ctx.beginPath();
        ctx.strokeStyle = stroke.color;
        ctx.lineWidth = stroke.size;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
        for (let i = 1; i < stroke.points.length; i++) {
          ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
        }
        ctx.stroke();
      }
    );
  }, [strokes, currentStroke, brushColor, brushSize]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  // Get position from event
  const getPosition = (e: React.MouseEvent | React.TouchEvent): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if ("touches" in e) {
      const touch = e.touches[0];
      return {
        x: (touch.clientX - rect.left) * scaleX,
        y: (touch.clientY - rect.top) * scaleY,
      };
    }

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleStart = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDrawing(true);
    const pos = getPosition(e);
    setCurrentStroke([pos]);
  };

  const handleMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getPosition(e);
    setCurrentStroke((prev) => [...prev, pos]);
  };

  const handleEnd = () => {
    if (currentStroke.length > 0) {
      setStrokes((prev) => [
        ...prev,
        { points: currentStroke, color: brushColor, size: brushSize },
      ]);
      setCurrentStroke([]);
      onMarkupChange?.(true);
    }
    setIsDrawing(false);
  };

  const handleClear = () => {
    setStrokes([]);
    setCurrentStroke([]);
    onMarkupChange?.(false);
    redraw();
  };

  const handleUndo = () => {
    setStrokes((prev) => {
      const newStrokes = prev.slice(0, -1);
      onMarkupChange?.(newStrokes.length > 0);
      return newStrokes;
    });
  };

  const handleExport = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob(
      (blob) => {
        if (blob) {
          onExport?.(blob);
        }
      },
      "image/jpeg",
      0.9
    );
  };

  // Expose export function
  useEffect(() => {
    if (strokes.length > 0) {
      handleExport();
    }
  }, [strokes]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full cursor-crosshair touch-none"
        onMouseDown={handleStart}
        onMouseMove={handleMove}
        onMouseUp={handleEnd}
        onMouseLeave={handleEnd}
        onTouchStart={handleStart}
        onTouchMove={handleMove}
        onTouchEnd={handleEnd}
      />

      {/* Controls */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
        <button
          onClick={handleUndo}
          disabled={strokes.length === 0}
          className="px-3 py-1.5 bg-[var(--color-bg-primary)]/80 backdrop-blur text-[var(--color-text-secondary)] text-sm rounded-lg border border-[var(--color-bg-elevated)] hover:border-[var(--color-accent)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          撤销
        </button>
        <button
          onClick={handleClear}
          disabled={strokes.length === 0}
          className="px-3 py-1.5 bg-[var(--color-bg-primary)]/80 backdrop-blur text-[var(--color-text-secondary)] text-sm rounded-lg border border-[var(--color-bg-elevated)] hover:border-[var(--color-error)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          清除
        </button>
      </div>

      {/* Instructions */}
      {strokes.length === 0 && !isDrawing && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-[var(--color-bg-primary)]/80 backdrop-blur rounded-lg">
          <p className="text-sm text-[var(--color-text-secondary)]">
            用手指或鼠标涂抹你感兴趣的区域
          </p>
        </div>
      )}
    </div>
  );
}
