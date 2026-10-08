const form = document.querySelector("#upload-form");
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const previewArea = document.querySelector("#preview-area");
const uploadedPreview = document.querySelector("#uploaded-preview");
const fileName = document.querySelector("#file-name");
const analyzeButton = document.querySelector("#analyze-button");
const status = document.querySelector("#status");
const result = document.querySelector("#result");

function setFile(file) {
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    status.textContent = "Please choose a JPG, PNG, or WEBP image.";
    analyzeButton.disabled = true;
    return;
  }
  uploadedPreview.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  previewArea.classList.remove("hidden");
  analyzeButton.disabled = false;
  status.textContent = "";
  result.classList.add("hidden");
}

input.addEventListener("change", () => setFile(input.files[0]));
["dragenter", "dragover"].forEach(event => dropZone.addEventListener(event, e => {
  e.preventDefault(); dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(event => dropZone.addEventListener(event, e => {
  e.preventDefault(); dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", e => {
  const [file] = e.dataTransfer.files;
  setFile(file);
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!input.files[0]) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  status.textContent = "";
  result.classList.add("hidden");
  try {
    const body = new FormData();
    body.append("image", input.files[0]);
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    document.querySelector("#processed-image").src = data.processed_image;
    document.querySelector("#face-info").textContent =
      `${data.detector} selected the clearest of ${data.face_count} detected face(s).`;
    document.querySelector("#prediction-label").textContent = data.label;
    document.querySelector("#confidence").textContent = `${(data.confidence * 100).toFixed(2)}%`;
    document.querySelector("#fake-probability").textContent = `${(data.fake_probability * 100).toFixed(2)}%`;
    document.querySelector("#result-model").textContent = data.model_name;
    document.querySelector("#confidence-bar").style.width = `${data.confidence * 100}%`;
    const resultCard = document.querySelector("#result-card");
    resultCard.classList.remove("fake", "real");
    resultCard.classList.add(data.label.toLowerCase());
    document.querySelector("#confidence-bar").style.background = data.label === "FAKE" ? "#dc2626" : "#16a34a";
    result.classList.remove("hidden");
  } catch (error) {
    status.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze image";
  }
});
