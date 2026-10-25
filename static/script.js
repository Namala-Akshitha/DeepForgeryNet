const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const preview = document.getElementById("preview");
const hint = document.getElementById("hint");
const go = document.getElementById("go");
const result = document.getElementById("result");
let current = null;

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  current = f;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false; hint.hidden = true; go.disabled = false; result.hidden = true;
}
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
["dragenter", "dragover"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => setFile(ev.dataTransfer.files[0]));

go.addEventListener("click", async () => {
  if (!current) return;
  go.disabled = true; go.textContent = "Checking…"; result.hidden = true;
  const fd = new FormData(); fd.append("image", current);
  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    result.hidden = false;
    if (data.error) { result.className = "result err"; result.textContent = data.error; }
    else if (data.label === "REAL") { result.className = "result real"; result.textContent = "✅ Real image"; }
    else { result.className = "result fake"; result.textContent = "⚠️ Fake image"; }
  } catch (e) {
    result.hidden = false; result.className = "result err"; result.textContent = "Could not reach the server.";
  }
  go.disabled = false; go.textContent = "Check image";
});
