"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import MarkupCanvas from "@/components/MarkupCanvas";

// Types
interface ExifData {
  location?: string;
  lat?: number;
  lon?: number;
  datetime?: string;
  camera?: string;
}

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface Insight {
  title: string;
  hook: string;
  explanation: string;
  visual_evidence?: string;
  fun_fact?: string;
  bounding_box?: BoundingBox;
  cropped_image?: string;  // base64 data URL
}

interface AnalysisResult {
  type: string;
  perception: string;
  response: {
    type: string;
    insights?: Insight[];
    theme?: string;
    invitation?: string;
    message?: string;
    suggestion?: string;
  };
  timings?: Record<string, number>;
  process?: {
    interest_level: string;
    discoveries_count: number;
  };
}

type AnalysisState = "idle" | "uploading" | "analyzing" | "complete" | "error";

type AnalysisMode = "auto" | "markup";

// Image modal state
interface ModalState {
  isOpen: boolean;
  imageSrc: string | null;
}

export default function Home() {
  const [state, setState] = useState<AnalysisState>("idle");
  const [image, setImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisPhase, setAnalysisPhase] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // EXIF and location state
  const [exifData, setExifData] = useState<ExifData | null>(null);
  const [manualLocation, setManualLocation] = useState<string>("");
  const [imageId, setImageId] = useState<string | null>(null);

  // Analysis mode state
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("auto");
  const [hasMarkup, setHasMarkup] = useState(false);
  const [markedImageBlob, setMarkedImageBlob] = useState<Blob | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{width: number, height: number} | null>(null);

  // Image modal state
  const [modal, setModal] = useState<ModalState>({ isOpen: false, imageSrc: null });

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      processFile(file);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  }, []);

  const processFile = async (file: File) => {
    setImageFile(file);
    setResult(null);
    setError(null);
    setExifData(null);
    setManualLocation("");
    setImageId(null);
    setHasMarkup(false);
    setMarkedImageBlob(null);
    setAnalysisMode("auto");

    // Preview image and get dimensions
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      if (!dataUrl) {
        setError("无法读取图片文件");
        setState("error");
        return;
      }
      setImage(dataUrl);

      // Get image dimensions
      const img = new window.Image();
      img.onload = () => {
        // Scale to fit container while maintaining aspect ratio
        const maxWidth = 800;
        const maxHeight = 600;
        let width = img.width;
        let height = img.height;

        if (width > maxWidth) {
          height = (height * maxWidth) / width;
          width = maxWidth;
        }
        if (height > maxHeight) {
          width = (width * maxHeight) / height;
          height = maxHeight;
        }

        setImageDimensions({ width: Math.round(width), height: Math.round(height) });
      };
      img.onerror = () => {
        console.error("Failed to load image for dimension calculation");
        // Set default dimensions as fallback
        setImageDimensions({ width: 400, height: 300 });
      };
      img.src = dataUrl;
    };
    reader.onerror = () => {
      console.error("FileReader error:", reader.error);
      setError("文件读取失败，请重新选择");
      setState("error");
    };
    reader.readAsDataURL(file);

    // Upload to get EXIF and image ID
    setState("uploading");
    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      setImageId(data.id);
      setExifData(data.exif || null);
      setState("idle");
    } catch (err) {
      console.error("Upload error:", err);
      setState("idle"); // Still allow manual analysis
    }
  };

  const [currentLayer, setCurrentLayer] = useState<number>(0);
  const [layerDetails, setLayerDetails] = useState<{[key: number]: {message: string, detail: string}}>({});

  const handleAnalyze = async () => {
    if (!imageFile) return;

    // In markup mode, require some markup
    if (analysisMode === "markup" && !hasMarkup) {
      setError("请先在图片上标记你感兴趣的区域");
      return;
    }

    setState("analyzing");
    setError(null);
    setAnalysisPhase("准备分析...");
    setCurrentLayer(0);
    setLayerDetails({});

    try {
      // Always upload for analysis (with markup if in markup mode)
      const formData = new FormData();

      if (analysisMode === "markup" && markedImageBlob) {
        // Use the marked image
        formData.append("image", markedImageBlob, "marked-image.jpg");
        formData.append("has_markup", "true");
      } else {
        // Use original image
        formData.append("image", imageFile);
        formData.append("has_markup", "false");
      }

      const uploadResponse = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.status}`);
      }

      const data = await uploadResponse.json();
      const currentImageId = data.id;
      setImageId(data.id);
      if (!exifData) {
        setExifData(data.exif || null);
      }

      setAnalysisPhase("连接分析服务...");

      // Build analysis URL with location params
      let analyzeUrl = `http://localhost:8000/api/analyze/${currentImageId}`;
      const params = new URLSearchParams();

      // Use EXIF location or manual location
      if (exifData?.lat && exifData?.lon) {
        params.set("lat", exifData.lat.toString());
        params.set("lon", exifData.lon.toString());
      } else if (manualLocation) {
        // TODO: Geocode manual location to lat/lon
        // For now, just log it
        console.log("Manual location:", manualLocation);
      }

      if (params.toString()) {
        analyzeUrl += `?${params.toString()}`;
      }

      // Connect to SSE stream
      const eventSource = new EventSource(analyzeUrl);

      eventSource.addEventListener("progress", (event) => {
        try {
          const data = JSON.parse(event.data);
          setCurrentLayer(data.layer || 0);
          setAnalysisPhase(data.message || "处理中...");
          if (data.layer) {
            setLayerDetails(prev => ({
              ...prev,
              [data.layer]: { message: data.message, detail: data.detail || "" }
            }));
          }
        } catch (parseError) {
          console.error("Failed to parse progress event:", parseError, event.data);
          // Non-fatal: continue processing
        }
      });

      eventSource.addEventListener("result", (event) => {
        try {
          const data = JSON.parse(event.data);
          eventSource.close();
          if (data.success) {
            setResult(data.data);
            setState("complete");
          } else {
            setError(data.error || "Analysis failed");
            setState("error");
          }
        } catch (parseError) {
          console.error("Failed to parse result event:", parseError, event.data);
          eventSource.close();
          setError("解析结果失败");
          setState("error");
        }
      });

      eventSource.addEventListener("error", (event) => {
        eventSource.close();
        // Check if it's a MessageEvent with data
        if (event instanceof MessageEvent && event.data) {
          try {
            const data = JSON.parse(event.data);
            setError(data.message || "分析出错");
          } catch {
            setError("分析出错");
          }
        } else {
          setError("连接中断");
        }
        setState("error");
      });

      eventSource.onerror = (err) => {
        // Only handle if not already in error/complete state
        if (state !== "error" && state !== "complete") {
          eventSource.close();
          console.error("EventSource error:", err);
          setError("服务连接失败，请检查网络后重试");
          setState("error");
        }
      };

    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setState("error");
    }
  };

  const handleReset = () => {
    setImage(null);
    setImageFile(null);
    setResult(null);
    setError(null);
    setState("idle");
    setAnalysisPhase("");
    setExifData(null);
    setManualLocation("");
    setImageId(null);
    setCurrentLayer(0);
    setLayerDetails({});
    setAnalysisMode("auto");
    setHasMarkup(false);
    setMarkedImageBlob(null);
    setImageDimensions(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const openImageModal = (src: string) => {
    setModal({ isOpen: true, imageSrc: src });
  };

  const closeImageModal = () => {
    setModal({ isOpen: false, imageSrc: null });
  };

  return (
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--color-bg-tertiary)] py-3 px-4 md:py-6 md:px-8">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2 md:gap-3">
            <div className="w-8 h-8 md:w-10 md:h-10 rounded-full bg-[var(--color-accent)] flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-[var(--color-bg-primary)] md:w-5 md:h-5">
                <circle cx="12" cy="12" r="10"/>
                <circle cx="12" cy="12" r="3"/>
                <line x1="12" y1="2" x2="12" y2="6"/>
                <line x1="12" y1="18" x2="12" y2="22"/>
                <line x1="2" y1="12" x2="6" y2="12"/>
                <line x1="18" y1="12" x2="22" y2="12"/>
              </svg>
            </div>
            <div>
              <h1 className="text-lg md:text-xl font-bold tracking-tight" style={{ fontFamily: "var(--font-syne)" }}>
                CityLens
              </h1>
              <p className="text-[10px] md:text-xs text-[var(--color-text-muted)] tracking-widest uppercase hidden sm:block">
                See Beyond the Surface
              </p>
            </div>
          </div>

          {(state === "complete" || image) && (
            <button
              onClick={handleReset}
              className="text-xs md:text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-accent)] transition-colors"
            >
              New Analysis
            </button>
          )}
        </div>
      </header>

      {/* Main Content - Responsive: stack on mobile, side-by-side on desktop */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Left Panel - Upload / Image */}
        <div className={`w-full md:w-1/2 border-b md:border-b-0 md:border-r border-[var(--color-bg-tertiary)] p-4 md:p-8 flex flex-col ${
          state === "complete" ? "hidden md:flex" : "flex"
        } ${!image ? "min-h-[50vh] md:min-h-0" : ""}`}>
          {!image ? (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 border-2 border-dashed border-[var(--color-bg-elevated)] rounded-lg flex flex-col items-center justify-center cursor-pointer hover:border-[var(--color-accent)] hover:bg-[var(--color-bg-secondary)] transition-all duration-300 group p-6 md:p-8"
            >
              <div className="w-16 h-16 md:w-20 md:h-20 rounded-full bg-[var(--color-bg-tertiary)] flex items-center justify-center mb-4 md:mb-6 group-hover:bg-[var(--color-bg-elevated)] transition-colors">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-[var(--color-text-secondary)] group-hover:text-[var(--color-accent)] transition-colors md:w-8 md:h-8">
                  <rect x="3" y="3" width="18" height="18" rx="2"/>
                  <circle cx="8.5" cy="8.5" r="1.5"/>
                  <path d="M21 15l-5-5L5 21"/>
                </svg>
              </div>
              <h2 className="text-xl md:text-2xl font-bold mb-2 text-center" style={{ fontFamily: "var(--font-syne)" }}>
                Drop your image here
              </h2>
              <p className="text-sm md:text-base text-[var(--color-text-secondary)] mb-2 md:mb-4">
                or click to browse
              </p>
              <p className="text-xs text-[var(--color-text-muted)]">
                JPEG, PNG up to 10MB
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          ) : (
            <div className="flex-1 flex flex-col min-h-0">
              {/* Mode Toggle */}
              {state === "idle" && (
                <div className="mb-3 md:mb-4 flex gap-2">
                  <button
                    onClick={() => setAnalysisMode("auto")}
                    className={`flex-1 py-2 px-3 md:px-4 rounded-lg text-xs md:text-sm font-medium transition-colors ${
                      analysisMode === "auto"
                        ? "bg-[var(--color-accent)] text-[var(--color-bg-primary)]"
                        : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elevated)]"
                    }`}
                  >
                    自动分析
                  </button>
                  <button
                    onClick={() => setAnalysisMode("markup")}
                    className={`flex-1 py-2 px-3 md:px-4 rounded-lg text-xs md:text-sm font-medium transition-colors ${
                      analysisMode === "markup"
                        ? "bg-[var(--color-accent)] text-[var(--color-bg-primary)]"
                        : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elevated)]"
                    }`}
                  >
                    标记探索
                  </button>
                </div>
              )}

              {/* Image Display / Markup Canvas */}
              <div className="flex-1 relative rounded-lg overflow-hidden bg-[var(--color-bg-secondary)] min-h-[200px] md:min-h-0">
                {analysisMode === "markup" && state === "idle" && imageDimensions ? (
                  <MarkupCanvas
                    imageUrl={image}
                    width={imageDimensions.width}
                    height={imageDimensions.height}
                    brushColor="rgba(255, 200, 0, 0.35)"
                    brushSize={35}
                    onMarkupChange={setHasMarkup}
                    onExport={setMarkedImageBlob}
                  />
                ) : (
                  <img
                    src={image}
                    alt="Uploaded"
                    className="w-full h-full object-contain animate-fade-in"
                    onError={(e) => {
                      console.error("Failed to load uploaded image");
                      setError("图片加载失败，请重新上传");
                      setState("error");
                    }}
                  />
                )}
              </div>

              {/* EXIF Info & Location Input */}
              {state !== "analyzing" && state !== "complete" && (
                <div className="mt-3 md:mt-4 p-3 md:p-4 bg-[var(--color-bg-secondary)] rounded-lg">
                  {state === "uploading" ? (
                    <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
                      <div className="w-4 h-4 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
                      <span className="text-xs md:text-sm">提取照片信息...</span>
                    </div>
                  ) : (
                    <div className="space-y-2 md:space-y-3">
                      {/* Show EXIF data if available */}
                      {exifData && (exifData.location || exifData.datetime || exifData.camera) && (
                        <div className="flex flex-wrap gap-2 md:gap-4 text-xs md:text-sm">
                          {exifData.location && (
                            <div className="flex items-center gap-1 md:gap-1.5 text-[var(--color-text-secondary)]">
                              <span>📍</span>
                              <span className="truncate max-w-[120px] md:max-w-none">{exifData.location}</span>
                            </div>
                          )}
                          {exifData.datetime && (
                            <div className="flex items-center gap-1 md:gap-1.5 text-[var(--color-text-secondary)]">
                              <span>📅</span>
                              <span>{exifData.datetime}</span>
                            </div>
                          )}
                          {exifData.camera && (
                            <div className="flex items-center gap-1 md:gap-1.5 text-[var(--color-text-muted)] hidden sm:flex">
                              <span>📷</span>
                              <span>{exifData.camera}</span>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Location input if no GPS */}
                      {!exifData?.location && (
                        <div>
                          <label className="block text-[10px] md:text-xs text-[var(--color-text-muted)] mb-1 md:mb-1.5">
                            拍摄地点 (可选，帮助提供更准确的分析)
                          </label>
                          <input
                            type="text"
                            value={manualLocation}
                            onChange={(e) => setManualLocation(e.target.value)}
                            placeholder="例如: 东京表参道、上海外滩..."
                            className="w-full px-3 py-2 bg-[var(--color-bg-tertiary)] border border-[var(--color-bg-elevated)] rounded-lg text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] focus:border-[var(--color-accent)] focus:outline-none transition-colors"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {state === "idle" && (
                <button
                  onClick={handleAnalyze}
                  disabled={analysisMode === "markup" && !hasMarkup}
                  className={`mt-3 md:mt-4 py-3 md:py-4 px-6 md:px-8 font-bold rounded-lg transition-colors text-base md:text-lg ${
                    analysisMode === "markup" && !hasMarkup
                      ? "bg-[var(--color-bg-tertiary)] text-[var(--color-text-muted)] cursor-not-allowed"
                      : "bg-[var(--color-accent)] text-[var(--color-bg-primary)] hover:bg-[var(--color-accent-light)]"
                  }`}
                  style={{ fontFamily: "var(--font-syne)" }}
                >
                  {analysisMode === "auto" ? "Analyze Scene" : hasMarkup ? "分析标记区域" : "请先标记区域"}
                </button>
              )}

              {state === "analyzing" && (
                <div className="mt-4 md:mt-6 py-3 md:py-4 px-6 md:px-8 bg-[var(--color-bg-tertiary)] rounded-lg flex items-center justify-center gap-2 md:gap-3">
                  <div className="w-4 h-4 md:w-5 md:h-5 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm md:text-base text-[var(--color-text-secondary)]">{analysisPhase}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className={`w-full md:w-1/2 p-4 md:p-8 overflow-y-auto ${
          state === "complete" ? "flex-1" : "flex-1 md:flex-none"
        }`}>
          {state === "idle" && !image && (
            <div className="h-full flex flex-col items-center justify-center text-center px-4 md:px-12 py-8">
              <div className="w-12 h-12 md:w-16 md:h-16 rounded-full border-2 border-[var(--color-bg-elevated)] flex items-center justify-center mb-4 md:mb-6">
                <span className="text-2xl md:text-3xl">🔍</span>
              </div>
              <h2 className="text-2xl md:text-3xl font-bold mb-3 md:mb-4" style={{ fontFamily: "var(--font-syne)" }}>
                What will you discover?
              </h2>
              <p className="text-sm md:text-base text-[var(--color-text-secondary)] leading-relaxed max-w-md">
                Upload a photo of any urban scene — a building facade, a street corner,
                a shop front — and let AI reveal the stories hidden in plain sight.
              </p>
            </div>
          )}

          {state === "idle" && image && (
            <div className="h-full flex flex-col items-center justify-center text-center px-4 md:px-12 py-8 md:py-0">
              <div className="w-12 h-12 md:w-16 md:h-16 rounded-full border-2 border-[var(--color-accent)] flex items-center justify-center mb-4 md:mb-6">
                <span className="text-2xl md:text-3xl">✨</span>
              </div>
              <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4" style={{ fontFamily: "var(--font-syne)" }}>
                Ready to analyze
              </h2>
              <p className="text-sm md:text-base text-[var(--color-text-secondary)]">
                Click the button to start the analysis
              </p>
            </div>
          )}

          {state === "analyzing" && (
            <div className="h-full flex flex-col items-center justify-center py-8 md:py-0">
              <div className="space-y-3 md:space-y-4 w-full max-w-sm md:max-w-md px-4 md:px-0">
                {[
                  { num: 1, name: "Triage", desc: "判断内容" },
                  { num: 2, name: "Observation", desc: "多视角观察" },
                  { num: 3, name: "Discovery", desc: "深入研究" },
                  { num: 4, name: "Synthesis", desc: "生成洞见" },
                  { num: 5, name: "Segmentation", desc: "提取区域" },
                ].map((layer) => {
                  const isActive = currentLayer === layer.num;
                  const isComplete = currentLayer > layer.num;
                  const layerInfo = layerDetails[layer.num];

                  return (
                    <div key={layer.num} className="flex items-start gap-3 md:gap-4">
                      <div className={`w-7 h-7 md:w-8 md:h-8 rounded-full flex items-center justify-center text-xs md:text-sm font-bold flex-shrink-0 transition-all duration-300 ${
                        isComplete ? "bg-[var(--color-success)] text-[var(--color-bg-primary)]" :
                        isActive ? "bg-[var(--color-accent)] text-[var(--color-bg-primary)]" :
                        "bg-[var(--color-bg-tertiary)] text-[var(--color-text-muted)]"
                      }`}>
                        {isComplete ? "✓" : layer.num}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1 md:gap-2">
                          <span className={`text-sm md:text-base font-medium ${isActive ? "text-[var(--color-accent)]" : isComplete ? "text-[var(--color-text-primary)]" : "text-[var(--color-text-muted)]"}`}>
                            {layer.name}
                          </span>
                          <span className="text-[10px] md:text-xs text-[var(--color-text-muted)]">{layer.desc}</span>
                        </div>
                        {isActive && (
                          <div className="mt-1.5 md:mt-2">
                            <div className="h-1 md:h-1.5 rounded-full animate-shimmer" />
                            {layerInfo?.detail && (
                              <p className="mt-1 text-[10px] md:text-xs text-[var(--color-text-secondary)] truncate">
                                {layerInfo.detail}
                              </p>
                            )}
                          </div>
                        )}
                        {isComplete && layerInfo && (
                          <p className="mt-1 text-[10px] md:text-xs text-[var(--color-text-muted)] truncate">
                            {layerInfo.message}
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-6 md:mt-8 text-[var(--color-accent)] text-xs md:text-sm font-medium px-4 text-center">
                {analysisPhase}
              </p>
            </div>
          )}

          {state === "error" && (
            <div className="h-full flex flex-col items-center justify-center text-center px-4 md:px-12 py-8">
              <div className="w-12 h-12 md:w-16 md:h-16 rounded-full bg-[var(--color-error)]/20 flex items-center justify-center mb-4 md:mb-6">
                <span className="text-2xl md:text-3xl">⚠️</span>
              </div>
              <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4 text-[var(--color-error)]" style={{ fontFamily: "var(--font-syne)" }}>
                Analysis Failed
              </h2>
              <p className="text-sm md:text-base text-[var(--color-text-secondary)] mb-4 md:mb-6">
                {error}
              </p>
              <button
                onClick={handleAnalyze}
                className="py-2 px-4 md:px-6 border border-[var(--color-text-secondary)] rounded-lg hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] transition-colors text-sm md:text-base"
              >
                Try Again
              </button>
            </div>
          )}

          {state === "complete" && result && (
            <div className="animate-fade-in-up">
              {/* Perception */}
              <div className="mb-6 md:mb-8 pb-6 md:pb-8 border-b border-[var(--color-bg-tertiary)]">
                <p className="text-[10px] md:text-xs text-[var(--color-accent)] uppercase tracking-widest mb-1.5 md:mb-2">
                  Perception
                </p>
                <p className="text-lg md:text-xl text-[var(--color-text-primary)] leading-relaxed italic">
                  &ldquo;{result.perception}&rdquo;
                </p>
              </div>

              {/* Theme */}
              {result.response.theme && (
                <div className="mb-6 md:mb-8">
                  <h2 className="text-2xl md:text-3xl font-bold mb-2" style={{ fontFamily: "var(--font-syne)" }}>
                    {result.response.theme}
                  </h2>
                </div>
              )}

              {/* Insights */}
              {result.response.insights && result.response.insights.length > 0 && (
                <div className="space-y-4 md:space-y-6 stagger-children">
                  {result.response.insights.map((insight, index) => (
                    <div
                      key={index}
                      className="insight-card p-4 md:p-6 bg-[var(--color-bg-secondary)] rounded-lg border border-[var(--color-bg-tertiary)] hover:border-[var(--color-accent)]/30 transition-colors"
                    >
                      <div className="flex items-start gap-3 md:gap-4">
                        <div className="w-7 h-7 md:w-8 md:h-8 rounded-full bg-[var(--color-accent)]/20 flex items-center justify-center flex-shrink-0">
                          <span className="text-[var(--color-accent)] font-bold text-xs md:text-sm">
                            {index + 1}
                          </span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3 className="text-base md:text-lg font-bold mb-1.5 md:mb-2 text-[var(--color-text-primary)]" style={{ fontFamily: "var(--font-syne)" }}>
                            {insight.title}
                          </h3>
                          {insight.hook && (
                            <p className="text-[var(--color-accent)] text-xs md:text-sm mb-2 md:mb-3 italic">
                              {insight.hook}
                            </p>
                          )}

                          {/* Cropped Image Display */}
                          {insight.cropped_image && (
                            <div className="mb-3 md:mb-4 relative group">
                              <div className="overflow-hidden rounded-lg border border-[var(--color-bg-tertiary)] bg-[var(--color-bg-tertiary)]">
                                <img
                                  src={insight.cropped_image}
                                  alt={insight.title}
                                  className="w-full h-auto max-h-36 md:max-h-48 object-contain cursor-zoom-in transition-transform group-hover:scale-105"
                                  onClick={() => openImageModal(insight.cropped_image!)}
                                  onError={(e) => {
                                    // Hide broken image
                                    (e.target as HTMLImageElement).style.display = 'none';
                                    console.error(`Failed to load cropped image for: ${insight.title}`);
                                  }}
                                />
                              </div>
                              <div className="absolute top-1.5 right-1.5 md:top-2 md:right-2 px-1.5 py-0.5 md:px-2 md:py-1 bg-black/50 rounded text-[10px] md:text-xs text-white/80">
                                🔍 点击放大
                              </div>
                            </div>
                          )}

                          <p className="text-sm md:text-base text-[var(--color-text-secondary)] leading-relaxed">
                            {insight.explanation}
                          </p>
                          {insight.fun_fact && (
                            <div className="mt-3 md:mt-4 p-2.5 md:p-3 bg-[var(--color-bg-tertiary)] rounded-lg">
                              <p className="text-xs md:text-sm">
                                <span className="text-[var(--color-accent)] mr-1.5 md:mr-2">💡</span>
                                <span className="text-[var(--color-text-secondary)]">{insight.fun_fact}</span>
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Simple message for non-deep results */}
              {result.response.message && (
                <div className="p-4 md:p-6 bg-[var(--color-bg-secondary)] rounded-lg">
                  <p className="text-sm md:text-base text-[var(--color-text-secondary)]">{result.response.message}</p>
                  {result.response.suggestion && (
                    <p className="mt-3 md:mt-4 text-xs md:text-sm text-[var(--color-accent)]">{result.response.suggestion}</p>
                  )}
                </div>
              )}

              {/* Invitation */}
              {result.response.invitation && (
                <div className="mt-6 md:mt-8 pt-6 md:pt-8 border-t border-[var(--color-bg-tertiary)]">
                  <p className="text-sm md:text-base text-[var(--color-text-secondary)] italic text-center">
                    {result.response.invitation}
                  </p>
                </div>
              )}

              {/* Stats */}
              {result.timings && (
                <div className="mt-6 md:mt-8 pt-6 md:pt-8 border-t border-[var(--color-bg-tertiary)]">
                  <div className="flex items-center justify-center gap-4 md:gap-8 text-[10px] md:text-xs text-[var(--color-text-muted)]">
                    <span>
                      Analysis time: {result.timings.total?.toFixed(1)}s
                    </span>
                    {result.process && (
                      <span>
                        Discoveries: {result.process.discoveries_count}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Image Modal */}
      {modal.isOpen && modal.imageSrc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 cursor-zoom-out"
          onClick={closeImageModal}
        >
          <img
            src={modal.imageSrc}
            alt="Enlarged view"
            className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
            onError={(e) => {
              console.error("Failed to load modal image");
              closeImageModal();
            }}
          />
          <button
            onClick={closeImageModal}
            className="absolute top-4 right-4 w-10 h-10 rounded-full bg-black/50 text-white flex items-center justify-center hover:bg-black/70 transition-colors"
          >
            ✕
          </button>
        </div>
      )}
    </main>
  );
}
