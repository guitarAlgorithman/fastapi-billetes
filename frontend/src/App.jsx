import { useState, useCallback } from "react";
import Cropper from "react-easy-crop";
import getCroppedImg from "./cropImage";

const API_URL = import.meta.env.VITE_API_URL; // viene de .env.local

function App() {
  const [file, setFile] = useState(null);           // archivo final (recortado)
  const [preview, setPreview] = useState(null);     // preview (original o recortado)
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Estados para recorte
  const [isCropping, setIsCropping] = useState(false);
  const [originalImageUrl, setOriginalImageUrl] = useState(null);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState(null);

  const onCropComplete = useCallback((_, croppedAreaPixels) => {
    setCroppedAreaPixels(croppedAreaPixels);
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;

    setResult(null);
    setError(null);

    const url = URL.createObjectURL(f);

    // Da igual si viene del cel o del PC: siempre entramos en modo crop
    setOriginalImageUrl(url);
    setPreview(url);
    setIsCropping(true);
    setFile(null); // aún no tenemos archivo final recortado
  };

  const aplicarRecorte = async () => {
    try {
      if (!originalImageUrl || !croppedAreaPixels) {
        setError("No se pudo obtener el área de recorte.");
        return;
      }

      const croppedBlob = await getCroppedImg(originalImageUrl, croppedAreaPixels);

      const croppedUrl = URL.createObjectURL(croppedBlob);
      setPreview(croppedUrl);

      const croppedFile = new File([croppedBlob], "billete_recortado.jpg", {
        type: "image/jpeg",
      });
      setFile(croppedFile);

      setIsCropping(false);
    } catch (err) {
      console.error(err);
      setError("Error al recortar la imagen.");
    }
  };

  const fileToBase64 = (f) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result); // data:image/jpeg;base64,...
      reader.onerror = (err) => reject(err);
      reader.readAsDataURL(f);
    });

  const handleSend = async () => {
    if (!file) {
      setError("Primero toma o sube una imagen, recorta el billete y luego analiza.");
      return;
    }
    if (!API_URL) {
      setError("No está configurada la URL de la API (VITE_API_URL).");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const base64 = await fileToBase64(file);

      const resp = await fetch(`${API_URL}/predict-billete-base64`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image_base64: base64 }),
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || `Error HTTP ${resp.status}`);
      }

      const data = await resp.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Error al contactar la API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "#0f172a",
        color: "#e5e7eb",
        padding: "1rem",
      }}
    >
      <div
        style={{
          maxWidth: "480px",
          width: "100%",
          background: "#111827",
          borderRadius: "16px",
          padding: "1.5rem",
          boxShadow: "0 20px 40px rgba(0,0,0,0.5)",
          border: "1px solid #1f2937",
        }}
      >
        <h1 style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>
          Clasificador de Billetes
        </h1>
        <p
          style={{
            fontSize: "0.9rem",
            color: "#9ca3af",
            marginBottom: "1rem",
          }}
        >
          Toma una foto o sube una imagen desde tu computador, recorta el billete
          y el modelo te dirá si es <b>apto</b> o <b>no apto</b>.
        </p>

        {/* Input de imagen (sirve móvil y PC) */}
        {!isCropping && (
          <input
            type="file"
            accept="image/*"
            capture="environment" // en móvil sugiere cámara; en PC lo ignoran
            onChange={handleFileChange}
            style={{ marginBottom: "1rem" }}
          />
        )}

        {/* Zona de recorte */}
        {isCropping && originalImageUrl && (
          <div style={{ marginBottom: "1rem" }}>
            <div
              style={{
                position: "relative",
                width: "100%",
                height: 300,
                background: "#000",
                borderRadius: "12px",
                overflow: "hidden",
              }}
            >
              <Cropper
                image={originalImageUrl}
                crop={crop}
                zoom={zoom}
                aspect={2.11} // proporción del billete chileno (148/70)
                onCropChange={setCrop}
                onZoomChange={setZoom}
                onCropComplete={onCropComplete}
              />
            </div>
            <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.5rem" }}>
              <input
                type="range"
                min={1}
                max={3}
                step={0.1}
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
                style={{ flex: 1 }}
              />
            </div>
            <button
              onClick={aplicarRecorte}
              style={{
                marginTop: "0.75rem",
                width: "100%",
                padding: "0.6rem",
                borderRadius: "999px",
                border: "none",
                fontSize: "0.95rem",
                fontWeight: "600",
                cursor: "pointer",
                background: "#22c55e",
                color: "#111827",
              }}
            >
              Usar recorte
            </button>
          </div>
        )}

        {/* Preview final (recortado) */}
        {preview && !isCropping && (
          <div
            style={{
              marginBottom: "1rem",
              borderRadius: "12px",
              overflow: "hidden",
              border: "1px solid #374151",
            }}
          >
            <img
              src={preview}
              alt="Preview billete"
              style={{
                width: "100%",
                display: "block",
                maxHeight: "260px",
                objectFit: "cover",
              }}
            />
          </div>
        )}

        {/* Botón de análisis */}
        <button
          onClick={handleSend}
          disabled={loading || !file || isCropping}
          style={{
            width: "100%",
            padding: "0.75rem",
            borderRadius: "999px",
            border: "none",
            fontSize: "1rem",
            fontWeight: "600",
            cursor: loading || !file || isCropping ? "not-allowed" : "pointer",
            background:
              loading || !file || isCropping ? "#4b5563" : "#22c55e",
            color: "#111827",
            marginBottom: "1rem",
          }}
        >
          {loading ? "Analizando..." : "Analizar billete"}
        </button>

        {error && ( 
          <div
            style={{
              background: "#7f1d1d",
              color: "#fee2e2",
              padding: "0.75rem",
              borderRadius: "8px",
              fontSize: "0.85rem",
              marginBottom: "0.5rem",
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {result && (
          <div
            style={{
              background: "#022c22",
              borderRadius: "12px",
              padding: "1rem",
              border: "1px solid #064e3b",
            }}
          >
            <div style={{ fontSize: "0.9rem", color: "#6ee7b7" }}>
              Resultado
            </div>
            <div
              style={{
                fontSize: "1.4rem",
                fontWeight: "700",
                marginTop: "0.25rem",
              }}
            >
              {result.clase?.toUpperCase()}
            </div>
            <div
              style={{
                marginTop: "0.5rem",
                fontSize: "0.9rem",
                color: "#a7f3d0",
              }}
            >
              Confianza:{" "}
              <b>{(result.confianza * 100).toFixed(2)}%</b>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
