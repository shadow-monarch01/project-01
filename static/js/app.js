/**
 * AI Hiring Intelligence - Frontend Client Engine
 * Handles real-time API communication, candidate selection, live LLM inference,
 * counterfactual perturbations, EFS scoring, interactive charts, and mitigation loops.
 */

let state = {
    activeDataset: "high_bias_hiring_dataset.csv",
    datasets: [],
    concepts: [],
    candidatePool: [],
    selectedCandidate: null,
    candidatePage: 1,
    candidatePageSize: 10,
    totalCandidatePages: 1,
    batchResults: null,
    inspectorFilter: "",
    ollamaStatus: { connected: false, models: [] }
};

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    initTheme();
    await checkOllamaStatus();
    await loadDatasets();
    await loadCandidates(1);
    await onDatasetChange();
    initPlaygroundValues();
}

// --- Theme Management ---
function initTheme() {
    const saved = localStorage.getItem("app-theme") || "dark";
    document.documentElement.setAttribute("data-theme", saved);
    updateThemeButtonUI(saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("app-theme", next);
    updateThemeButtonUI(next);
}

function updateThemeButtonUI(theme) {
    const icon = document.getElementById("theme-icon");
    const text = document.getElementById("theme-text");
    if (theme === "light") {
        if (icon) icon.textContent = "🌙";
        if (text) text.textContent = "Dark Mode";
    } else {
        if (icon) icon.textContent = "☀️";
        if (text) text.textContent = "Light Mode";
    }
}

// --- Tab Navigation ---
function switchTab(tabId) {
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));

    const pane = document.getElementById(tabId);
    if (pane) pane.classList.add("active");

    const btn = Array.from(document.querySelectorAll(".tab-btn")).find(b => b.getAttribute("onclick") && b.getAttribute("onclick").includes(tabId));
    if (btn) btn.classList.add("active");
}

// --- API & Ollama Status ---
async function checkOllamaStatus(isManual = false) {
    const urlInput = document.getElementById("ollama-url");
    const hostUrl = urlInput ? urlInput.value.trim() : "http://127.0.0.1:11434";
    const badge = document.getElementById("ollama-status-badge");
    const text = document.getElementById("ollama-status-text");

    try {
        const resp = await fetch(`/api/ollama/status?url=${encodeURIComponent(hostUrl)}`);
        const data = await resp.json();
        state.ollamaStatus = data;

        if (data.connected) {
            if (badge) {
                badge.style.background = "rgba(16, 185, 129, 0.15)";
                badge.style.color = "#10b981";
            }
            if (text) text.textContent = `Online (${data.total_models} models detected)`;
            populateOllamaModels(data.models);
        } else {
            if (badge) {
                badge.style.background = "rgba(239, 68, 68, 0.15)";
                badge.style.color = "#ef4444";
            }
            if (text) text.textContent = "Offline (Simulation fallback active)";
        }
    } catch (e) {
        if (badge) {
            badge.style.background = "rgba(239, 68, 68, 0.15)";
            badge.style.color = "#ef4444";
        }
        if (text) text.textContent = "Service Unreachable";
    }
}

function populateOllamaModels(models) {
    const select = document.getElementById("ollama-model-select");
    if (!select || !models || !models.length) return;
    select.innerHTML = "";
    models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m.includes("qwen3.5") || m.includes("llama3")) opt.selected = true;
        select.appendChild(opt);
    });
}

function toggleApiConfig() {
    const mode = document.getElementById("select-mode").value;
    const ollamaBox = document.getElementById("ollama-config-container");
    const cloudBox = document.getElementById("api-config-container");

    if (ollamaBox) ollamaBox.classList.toggle("hidden", mode !== "Local Ollama Mode");
    if (cloudBox) cloudBox.classList.toggle("hidden", mode !== "Real LLM API Mode");
}

// --- Datasets & Concepts Discovery ---
async function loadDatasets() {
    try {
        const resp = await fetch("/api/datasets");
        const data = await resp.json();
        state.datasets = data.datasets || [];
        state.activeDataset = data.active || "high_bias_hiring_dataset.csv";

        const select = document.getElementById("select-dataset");
        if (select) {
            select.innerHTML = "";
            state.datasets.forEach(d => {
                const opt = document.createElement("option");
                opt.value = d.filename;
                opt.textContent = `${d.filename} (${d.rows} records)`;
                if (d.filename === state.activeDataset) opt.selected = true;
                select.appendChild(opt);
            });
        }
    } catch (e) {
        console.error("Error loading datasets:", e);
    }
}

async function onDatasetChange() {
    const select = document.getElementById("select-dataset");
    if (!select) return;
    state.activeDataset = select.value;
    await discoverConcepts();
    await loadCandidates(1);
}

async function discoverConcepts() {
    try {
        const resp = await fetch(`/api/dataset_concepts?dataset_name=${encodeURIComponent(state.activeDataset)}`);
        const data = await resp.json();
        state.concepts = data.concepts || [];

        const selectConcept = document.getElementById("select-concept");
        if (selectConcept && state.concepts.length > 0) {
            selectConcept.innerHTML = "";
            state.concepts.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c.id;
                opt.textContent = c.display_name;
                selectConcept.appendChild(opt);
            });
        }
        await onConceptChange();
    } catch (e) {
        console.error("Error discovering concepts:", e);
    }
}

async function onConceptChange() {
    const selectConcept = document.getElementById("select-concept");
    if (!selectConcept) return;
    const concept = selectConcept.value;

    try {
        const resp = await fetch(`/api/concept_options?dataset_name=${encodeURIComponent(state.activeDataset)}&concept=${encodeURIComponent(concept)}`);
        const data = await resp.json();

        const selA = document.getElementById("select-val-a");
        const selB = document.getElementById("select-val-b");
        if (!selA || !selB) return;

        selA.innerHTML = "";
        selB.innerHTML = "";

        const values = data.available_values || ["Group A", "Group B"];
        values.forEach((v, idx) => {
            const optA = document.createElement("option");
            optA.value = v; optA.textContent = v;
            if (data.default_pair && data.default_pair.val_a === v) optA.selected = true;
            else if (idx === 0) optA.selected = true;
            selA.appendChild(optA);

            const optB = document.createElement("option");
            optB.value = v; optB.textContent = v;
            if (data.default_pair && data.default_pair.val_b === v) optB.selected = true;
            else if (idx === 1 || (idx === 0 && values.length === 1)) optB.selected = true;
            selB.appendChild(optB);
        });
    } catch (e) {
        console.error("Error fetching concept options:", e);
    }
}

async function uploadCustomDataset() {
    const fileInput = document.getElementById("file-upload");
    if (!fileInput || !fileInput.files.length) {
        alert("Please select a valid CSV file first.");
        return;
    }
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const resp = await fetch("/api/upload_dataset", { method: "POST", body: formData });
        const data = await resp.json();
        if (resp.ok) {
            alert(`Uploaded '${data.filename}' successfully! (${data.rows} candidate records)`);
            await loadDatasets();
            const select = document.getElementById("select-dataset");
            if (select) select.value = data.filename;
            await onDatasetChange();
        } else {
            alert("Upload failed: " + (data.detail || "Unknown error"));
        }
    } catch (e) {
        alert("Upload error: " + e.message);
    }
}

// --- Candidate Pool & Interactive Row Selection ---
async function loadCandidates(page = 1) {
    state.candidatePage = page;
    const searchInput = document.getElementById("candidate-search");
    const query = searchInput ? searchInput.value.trim() : "";

    try {
        const resp = await fetch(`/api/candidates?dataset_name=${encodeURIComponent(state.activeDataset)}&page=${page}&page_size=${state.candidatePageSize}&search=${encodeURIComponent(query)}`);
        const data = await resp.json();
        state.candidatePool = data.candidates || [];
        state.totalCandidatePages = data.total_pages || 1;

        const sub = document.getElementById("candidate-pool-sub");
        if (sub) sub.textContent = `Showing ${state.candidatePool.length} of ${data.total_records} candidates in '${data.dataset_name}'`;

        renderCandidateTable(data.columns || [], state.candidatePool);
        renderCandidatePagination();
        populatePlaygroundAutofill(state.candidatePool);

        // If a candidate is currently selected, re-highlight and refresh card
        if (state.selectedCandidate) {
            const found = state.candidatePool.find(c => c.candidate_id === state.selectedCandidate.candidate_id);
            if (found) {
                selectCandidateById(found.candidate_id);
            }
        }
    } catch (e) {
        console.error("Error loading candidate pool:", e);
    }
}

function onCandidateSearch() {
    loadCandidates(1);
}

function setupCandidateTableEventDelegation() {
    const table = document.getElementById("candidate-pool-table");
    if (!table || table.dataset.delegated === "true") return;
    table.dataset.delegated = "true";

    table.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        const tr = e.target.closest("tr");
        if (!tr || !tr.dataset.candId) return;

        const candId = tr.dataset.candId;
        selectCandidateById(candId);

        if (btn) {
            e.stopPropagation();
            sendSelectedToPlayground();
        }
    });
}

function renderCandidateTable(columns, candidates) {
    const thead = document.getElementById("candidate-pool-thead");
    const tbody = document.getElementById("candidate-pool-tbody");
    if (!thead || !tbody) return;

    setupCandidateTableEventDelegation();

    if (!candidates.length) {
        thead.innerHTML = "";
        tbody.innerHTML = `<tr><td colspan="12" class="text-center" style="padding: 24px; color: var(--text-muted);">No candidate profiles found matching search criteria.</td></tr>`;
        return;
    }

    // Display primary columns
    const priorityCols = ["candidate_id", "name", "gender", "language", "university_tier", "education", "experience_years", "interview_score", "expected_role", "cluster"];
    const displayCols = priorityCols.filter(c => columns.includes(c) || (candidates[0] && candidates[0][c] !== undefined));
    if (displayCols.length < 4) {
        columns.slice(0, 7).forEach(c => { if (!displayCols.includes(c)) displayCols.push(c); });
    }

    thead.innerHTML = `<tr>${displayCols.map(c => `<th>${c.replace('_', ' ').toUpperCase()}</th>`).join("")}<th>ACTION</th></tr>`;

    tbody.innerHTML = "";
    candidates.forEach((cand, idx) => {
        const tr = document.createElement("tr");
        const candId = cand.candidate_id || `C${idx+1}`;
        tr.dataset.candId = candId;
        tr.dataset.index = idx;
        tr.className = "candidate-row";
        tr.style.cursor = "pointer";
        tr.style.transition = "all 0.15s ease";
        tr.setAttribute("onclick", `selectCandidateById('${candId}')`);
        
        if (state.selectedCandidate && (state.selectedCandidate.candidate_id === candId)) {
            tr.classList.add("selected-row");
            tr.style.backgroundColor = "rgba(6, 182, 212, 0.22)";
            tr.style.borderLeft = "4px solid var(--cyan)";
        }

        const tds = displayCols.map(c => {
            let val = cand[c] !== undefined ? cand[c] : "-";
            if (c === "cluster") return `<td><span class="badge" style="background: rgba(99, 102, 241, 0.2); color: var(--primary); font-weight: 700;">Cluster ${val}</span></td>`;
            if (c === "candidate_id") return `<td style="font-weight: 800; color: var(--cyan);">${val}</td>`;
            if (c === "name") return `<td style="font-weight: 700; color: var(--text-main);">${val}</td>`;
            if (c === "interview_score") return `<td style="font-weight: 700; color: #10b981;">${val}</td>`;
            return `<td>${val}</td>`;
        }).join("");

        tr.innerHTML = `${tds}<td><button class="btn-primary" style="padding: 4px 10px; font-size: 11.5px; border-radius: 4px; box-shadow: 0 2px 6px rgba(99,102,241,0.3);" onclick="event.stopPropagation(); selectCandidateById('${candId}'); sendSelectedToPlayground();">🔬 Test</button></td>`;

        tbody.appendChild(tr);
    });
}

function selectCandidateById(candId) {
    if (!candId) return;
    const cand = state.candidatePool.find(c => String(c.candidate_id).trim() === String(candId).trim())
                 || state.candidatePool.find(c => String(c.candidate_id).toLowerCase() === String(candId).toLowerCase())
                 || state.candidatePool[0];
    if (!cand) return;
    state.selectedCandidate = cand;

    // Highlight row across entire table
    document.querySelectorAll("#candidate-pool-tbody tr").forEach(r => {
        const isMatch = (r.dataset.candId === cand.candidate_id);
        r.classList.toggle("selected-row", isMatch);
        if (isMatch) {
            r.style.backgroundColor = "rgba(6, 182, 212, 0.22)";
            r.style.borderLeft = "4px solid var(--cyan)";
        } else {
            r.style.backgroundColor = "";
            r.style.borderLeft = "";
        }
    });

    // Populate all 13 attributes into the Selected Candidate Card
    renderSelectedCandidateCard(cand);
}

function selectCandidateRow(idx) {
    if (typeof idx === "number" && state.candidatePool[idx]) {
        selectCandidateById(state.candidatePool[idx].candidate_id);
    } else if (typeof idx === "string") {
        selectCandidateById(idx);
    }
}

function renderSelectedCandidateCard(cand) {
    if (!cand) return;
    const card = document.getElementById("selected-candidate-card");
    if (card) {
        card.classList.remove("hidden");
        card.style.display = "block";
        
        const setVal = (id, val, fallback = "-") => {
            const el = document.getElementById(id);
            if (el) el.textContent = (val !== undefined && val !== null && String(val).trim() !== "") ? val : fallback;
        };

        setVal("sel-cand-name", `${cand.name || "Candidate"} (${cand.candidate_id || "ID"})`);
        setVal("sel-attr-id", cand.candidate_id);
        setVal("sel-attr-role", cand.expected_role || cand.role || "Software Engineer");
        setVal("sel-attr-exp", `${cand.experience_years !== undefined ? cand.experience_years : (cand.experience || 5)} yrs`);
        setVal("sel-attr-score", `${cand.interview_score !== undefined ? cand.interview_score : (cand.score || 85)} / 100`);
        setVal("sel-attr-gender", cand.gender || cand.sex || "-");
        setVal("sel-attr-lang", cand.language || cand.english || "-");
        setVal("sel-attr-age", cand.age_group || cand.age || "-");
        setVal("sel-attr-edu", cand.education || cand.degree || "-");
        setVal("sel-attr-tier", cand.university_tier || cand.tier || "Tier-2 Regional");
        setVal("sel-attr-certs", cand.certifications_count !== undefined ? `${cand.certifications_count} Certifications` : "0 Certifications");
        setVal("sel-attr-salary", cand.previous_salary ? `$${Number(cand.previous_salary).toLocaleString()}` : "-");
        setVal("sel-attr-cluster", cand.cluster !== undefined ? `Cluster ${cand.cluster}` : "Unclustered");
        setVal("sel-attr-skills", cand.technical_skills || cand.skills || "Python; Distributed Systems; SQL; Docker");
    }
}

function sendSelectedToPlayground() {
    if (!state.selectedCandidate) {
        if (state.candidatePool.length > 0) {
            selectCandidateById(state.candidatePool[0].candidate_id);
        } else {
            return;
        }
    }
    const cand = state.selectedCandidate;
    setPlaygroundCandidate(cand);
    switchTab("tab-playground");
}

function renderCandidatePagination() {
    const bar = document.getElementById("candidate-pagination");
    if (!bar) return;
    bar.innerHTML = `
        <button ${state.candidatePage <= 1 ? "disabled" : ""} onclick="loadCandidates(${state.candidatePage - 1})">◀ Prev</button>
        <span style="font-size: 12px; align-self: center; color: var(--text-muted);">Page ${state.candidatePage} of ${state.totalCandidatePages}</span>
        <button ${state.candidatePage >= state.totalCandidatePages ? "disabled" : ""} onclick="loadCandidates(${state.candidatePage + 1})">Next ▶</button>
    `;
}

async function runClustering() {
    try {
        const resp = await fetch(`/api/cluster?dataset_name=${encodeURIComponent(state.activeDataset)}&n_clusters=3`, { method: "POST" });
        const data = await resp.json();
        const banner = document.getElementById("clustering-summary");
        if (banner) {
            banner.classList.remove("hidden");
            banner.style.background = "var(--bg-surface-elevated)";
            banner.style.border = "1px solid var(--primary-glow)";
            banner.style.padding = "12px 16px";
            banner.style.borderRadius = "var(--radius-sm)";
            banner.style.marginBottom = "14px";
            
            const distBadges = Object.entries(data.distribution).map(([cl, count]) => `
                <span class="badge" style="background: rgba(99, 102, 241, 0.2); color: var(--primary); padding: 4px 10px; margin-right: 8px;">
                    Cluster ${cl}: ${count} Candidates
                </span>
            `).join("");

            banner.innerHTML = `
                <div class="flex-between">
                    <div>
                        <strong style="color: var(--text-main);">🔍 TF-IDF + K-Means Clustering Results (k=3):</strong>
                        <div style="margin-top: 6px;">${distBadges}</div>
                    </div>
                    <span style="font-size: 11.5px; color: var(--text-muted);">Partitioned across semantic resume profiles</span>
                </div>
            `;
        }
        await loadCandidates(state.candidatePage);
    } catch (e) {
        alert("Clustering error: " + e.message);
    }
}

// --- Counterfactual Playground Engine ---
function populatePlaygroundAutofill(candidates) {
    const select = document.getElementById("play-candidate-select");
    if (!select) return;
    select.innerHTML = '<option value="">-- Select Candidate to Auto-fill --</option>';
    candidates.forEach((c, i) => {
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = `${c.name || "Candidate"} (${c.expected_role || c.role || "Engineer"} - ${c.experience_years || c.experience || 4}y)`;
        select.appendChild(opt);
    });
}

function autofillPlaygroundFromDataset() {
    const select = document.getElementById("play-candidate-select");
    if (!select || select.value === "") return;
    const cand = state.candidatePool[parseInt(select.value)];
    if (cand) setPlaygroundCandidate(cand);
}

function setSelectValueSafely(selectId, val) {
    const el = document.getElementById(selectId);
    if (!el || val === undefined || val === null) return;
    const strVal = String(val).trim();
    let found = false;
    for (let opt of el.options) {
        if (opt.value.toLowerCase() === strVal.toLowerCase()) {
            opt.selected = true;
            found = true;
            break;
        }
    }
    if (!found && strVal) {
        const newOpt = document.createElement("option");
        newOpt.value = strVal;
        newOpt.textContent = strVal;
        newOpt.selected = true;
        el.appendChild(newOpt);
    }
}

function setPlaygroundCandidate(cand) {
    if (document.getElementById("play-name")) document.getElementById("play-name").value = cand.name || cand.candidate_name || "Candidate";
    if (document.getElementById("play-role")) document.getElementById("play-role").value = cand.expected_role || cand.role || "Software Engineer";
    
    setSelectValueSafely("play-gender", cand.gender || cand.sex || "Female");
    setSelectValueSafely("play-language", cand.language || cand.english || "Basic");
    setSelectValueSafely("play-religion", cand.religion || cand.faith || "Hindu");
    setSelectValueSafely("play-ethnicity", cand.ethnicity || cand.race || "South Asian");
    setSelectValueSafely("play-age", cand.age_group || cand.age || "25-34");
    setSelectValueSafely("play-education", cand.education || cand.degree || "Tier-2 Regional");
    
    if (document.getElementById("play-exp")) document.getElementById("play-exp").value = cand.experience_years !== undefined ? cand.experience_years : (cand.experience || 5);
    if (document.getElementById("play-interview")) document.getElementById("play-interview").value = cand.interview_score !== undefined ? cand.interview_score : (cand.score || 85);
    if (document.getElementById("play-skills")) document.getElementById("play-skills").value = cand.technical_skills || cand.skills || "Python; Distributed Systems; SQL; Docker";

    // Configure default perturbation
    const conceptSel = document.getElementById("play-perturb-concept");
    if (conceptSel) {
        conceptSel.value = "gender";
        onPlaygroundConceptChange();
        const candGender = String(cand.gender || "Female").trim().toLowerCase();
        const targetGender = candGender === "female" ? "Male" : "Female";
        setSelectValueSafely("play-perturb-val", targetGender);
    }

    updateCounterfactualPreview();
}

function initPlaygroundValues() {
    onPlaygroundConceptChange();
    updateCounterfactualPreview();
}

function onPlaygroundConceptChange() {
    const conceptSel = document.getElementById("play-perturb-concept");
    const valSel = document.getElementById("play-perturb-val");
    if (!conceptSel || !valSel) return;

    const concept = conceptSel.value;
    valSel.innerHTML = "";

    const optionsMap = {
        "language": ["Fluent", "Basic", "Native", "Intermediate"],
        "gender": ["Male", "Female"],
        "religion": ["Christian", "Hindu", "Muslim", "Sikh", "Jewish", "None"],
        "ethnicity": ["White", "South Asian", "Black", "Hispanic", "East Asian"],
        "age": ["18-24", "25-34", "35-44", "45-54", "55+"],
        "education": ["Tier-1 Elite", "Tier-2 Regional", "Community College", "Bootcamp"]
    };

    const vals = optionsMap[concept] || ["Option A", "Option B"];
    vals.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        valSel.appendChild(opt);
    });

    updateCounterfactualPreview();
}

function updateCounterfactualPreview() {
    const box = document.getElementById("cf-preview-content");
    if (!box) return;

    const name = document.getElementById("play-name").value;
    const role = document.getElementById("play-role").value;
    const exp = document.getElementById("play-exp").value;
    const score = document.getElementById("play-interview").value;
    const concept = document.getElementById("play-perturb-concept").value;
    const targetVal = document.getElementById("play-perturb-val").value;

    box.innerHTML = `
        <strong>Profile:</strong> ${name} | <strong>Role:</strong> ${role}<br>
        <strong>Qualifications:</strong> ${exp} yrs experience | Technical Rating: ${score}/100<br>
        <strong>Perturbation:</strong> <span style="color: var(--cyan); font-weight: 700;">${concept.toUpperCase()} modified to '${targetVal}'</span> (all other factors invariant).
    `;
}

function getPlaygroundOriginalProfile() {
    return {
        "name": document.getElementById("play-name").value,
        "expected_role": document.getElementById("play-role").value,
        "gender": document.getElementById("play-gender").value,
        "language": document.getElementById("play-language").value,
        "religion": document.getElementById("play-religion").value,
        "ethnicity": document.getElementById("play-ethnicity").value,
        "age_group": document.getElementById("play-age").value,
        "education": document.getElementById("play-education").value,
        "experience_years": parseFloat(document.getElementById("play-exp").value) || 5,
        "interview_score": parseFloat(document.getElementById("play-interview").value) || 85,
        "technical_skills": document.getElementById("play-skills").value
    };
}

async function evaluatePlaygroundPair(mitigation = false) {
    const origProfile = getPlaygroundOriginalProfile();
    const concept = document.getElementById("play-perturb-concept").value;
    const targetVal = document.getElementById("play-perturb-val").value;

    const mode = document.getElementById("select-mode").value;
    const decType = document.getElementById("select-decision-type").value;
    const model = document.getElementById("ollama-model-select") ? document.getElementById("ollama-model-select").value : "qwen3.5:4b";

    // 1. Generate Counterfactual Profile via API
    let modProfile = { ...origProfile };
    try {
        const cfResp = await fetch("/api/counterfactual", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ candidate_data: origProfile, concept: concept, target_value: targetVal })
        });
        const cfData = await cfResp.json();
        modProfile = cfData.counterfactual_profile || modProfile;
    } catch (e) {
        modProfile[concept] = targetVal;
    }

    // 2. Evaluate Both Profiles via API
    try {
        const [resOrig, resMod] = await Promise.all([
            fetch("/api/evaluate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ candidate_data: origProfile, decision_type: decType, mode: mode, mitigation: mitigation, model_name: model })
            }).then(r => r.json()),
            fetch("/api/evaluate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ candidate_data: modProfile, decision_type: decType, mode: mode, mitigation: mitigation, model_name: model })
            }).then(r => r.json())
        ]);

        // 3. Compute EFS via API
        const efsResp = await fetch("/api/efs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                decision_orig: resOrig.decision,
                decision_mod: resMod.decision,
                explanation: resOrig.explanation,
                concept: concept,
                decision_type: decType
            })
        });
        const efsData = await efsResp.json();

        // 4. Render UI
        document.getElementById("play-empty-state").classList.add("hidden");
        document.getElementById("play-results-grid").classList.remove("hidden");
        document.getElementById("play-diagnostics-bar").classList.remove("hidden");

        const scoreOrigStr = resOrig.score !== undefined ? ` (Score: ${resOrig.score})` : '';
        const scoreModStr = resMod.score !== undefined ? ` (Score: ${resMod.score})` : '';

        document.getElementById("play-out-dec-orig").textContent = `${resOrig.decision}${scoreOrigStr}`;
        document.getElementById("play-out-exp-orig").textContent = resOrig.explanation;

        document.getElementById("play-out-dec-mod").textContent = `${resMod.decision}${scoreModStr}`;
        document.getElementById("play-out-exp-mod").textContent = resMod.explanation;

        const shiftBadge = efsData.is_changed 
            ? '<span style="color: #ef4444; font-weight: 800;">YES (Decision Disparity)</span>' 
            : '<span style="color: #10b981; font-weight: 800;">NO (Decision Invariant)</span>';
        document.getElementById("play-stat-shift").innerHTML = shiftBadge;
        document.getElementById("play-stat-verbal").textContent = efsData.is_verbalized ? "YES (Mentioned)" : "NO (Unmentioned)";
        document.getElementById("play-stat-efs").textContent = `${efsData.faithfulness_score} / 100`;
        
        const timeNow = new Date().toLocaleTimeString();
        document.getElementById("play-stat-diag").innerHTML = `
            <strong>Model:</strong> ${model} | <strong>Mode:</strong> ${mode} | <strong>Time:</strong> ${timeNow}<br>
            <strong>Diagnosis:</strong> ${efsData.diagnosis}
        `;

        const qBadge = document.getElementById("play-quadrant-badge");
        if (qBadge) {
            qBadge.classList.remove("hidden");
            qBadge.textContent = efsData.quadrant;
            if (efsData.quadrant_code === "Q1_HIDDEN") {
                qBadge.style.background = "rgba(239, 68, 68, 0.2)";
                qBadge.style.color = "#ef4444";
            } else if (efsData.quadrant_code === "Q3_INVARIANT") {
                qBadge.style.background = "rgba(16, 185, 129, 0.2)";
                qBadge.style.color = "#10b981";
            } else {
                qBadge.style.background = "rgba(245, 158, 11, 0.2)";
                qBadge.style.color = "#f59e0b";
            }
        }
    } catch (e) {
        alert("Evaluation error: " + e.message);
    }
}

// --- 1-Click Benchmark Presets ---
async function runPresetLanguageRegression() {
    const setSelect = document.getElementById("select-dataset");
    if (setSelect) setSelect.value = "high_bias_hiring_dataset.csv";
    await onDatasetChange();

    document.getElementById("select-decision-type").value = "regression";
    document.getElementById("select-concept").value = "language";
    await onConceptChange();

    document.getElementById("select-val-a").value = "Fluent";
    document.getElementById("select-val-b").value = "Basic";

    await runBatchAnalysis();
    switchTab("tab-analytics");
}

async function runPresetGenderBinary() {
    const setSelect = document.getElementById("select-dataset");
    if (setSelect) setSelect.value = "high_bias_hiring_dataset.csv";
    await onDatasetChange();

    document.getElementById("select-decision-type").value = "binary";
    document.getElementById("select-concept").value = "gender";
    await onConceptChange();

    document.getElementById("select-val-a").value = "Female";
    document.getElementById("select-val-b").value = "Male";

    await runBatchAnalysis();
    switchTab("tab-analytics");
}

// --- Batch Analysis & Visual Dashboard ---
async function runBatchAnalysis() {
    const btn = document.getElementById("btn-run-analysis");
    if (btn) { btn.disabled = true; btn.textContent = "⏳ Analyzing Candidate Pairs..."; }

    const payload = {
        dataset_name: state.activeDataset,
        concept: document.getElementById("select-concept").value,
        val_a: document.getElementById("select-val-a").value,
        val_b: document.getElementById("select-val-b").value,
        decision_type: document.getElementById("select-decision-type").value,
        mode: document.getElementById("select-mode").value,
        model_name: document.getElementById("ollama-model-select") ? document.getElementById("ollama-model-select").value : "qwen3.5:4b"
    };

    try {
        const resp = await fetch("/api/run_batch_analysis", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Analysis failed");

        state.batchResults = data;
        renderAnalyticsDashboard(data);
    } catch (e) {
        alert("Batch analysis error: " + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = "🚀 Run Bias & Faithfulness Analysis"; }
    }
}

function renderAnalyticsDashboard(data) {
    const m = data.metrics || {};
    const f = data.faithfulness_summary || {};
    const qCounts = f.quadrant_counts || { Q1_HIDDEN: 0, Q2_TRANSPARENT: 0, Q3_INVARIANT: 0, Q4_SUPERFLUOUS: 0 };

    document.getElementById("metric-effect-title").textContent = m.effect_name || "Discrepancy Rate";
    document.getElementById("metric-effect-val").textContent = `${m.effect_value}${data.params.decision_type === "regression" ? " pts" : "%"}`;
    document.getElementById("metric-effect-sub").textContent = `${data.params.val_a} vs ${data.params.val_b}`;

    document.getElementById("metric-efs-val").textContent = `${m.mean_faithfulness} / 100`;
    document.getElementById("metric-deception-val").textContent = `${m.deception_rate}%`;
    document.getElementById("metric-pval-val").textContent = m.p_value !== null ? m.p_value : "N/A";
    document.getElementById("metric-test-name").textContent = m.test_method || "Hypothesis Test";

    // Quadrants
    document.getElementById("count-q1").textContent = qCounts.Q1_HIDDEN || 0;
    document.getElementById("count-q2").textContent = qCounts.Q2_TRANSPARENT || 0;
    document.getElementById("count-q3").textContent = qCounts.Q3_INVARIANT || 0;
    document.getElementById("count-q4").textContent = qCounts.Q4_SUPERFLUOUS || 0;

    // Progress Donut / Bar
    const total = f.total_evaluated || 1;
    const q1Pct = Math.round(((qCounts.Q1_HIDDEN || 0) / total) * 100);
    const q3Pct = 100 - q1Pct;
    const barQ1 = document.getElementById("bar-q1");
    const barQ3 = document.getElementById("bar-q3");
    if (barQ1) { barQ1.style.width = `${q1Pct}%`; barQ1.textContent = `Q1: ${q1Pct}%`; }
    if (barQ3) { barQ3.style.width = `${q3Pct}%`; barQ3.textContent = `Q3: ${q3Pct}%`; }

    renderInspectorTable(data.candidate_details || []);
}

function renderInspectorTable(details) {
    const tbody = document.getElementById("inspector-tbody");
    if (!tbody) return;

    if (!details.length) {
        tbody.innerHTML = `<tr><td colspan="10" class="text-center">No candidate details available.</td></tr>`;
        return;
    }

    tbody.innerHTML = details.map((c, i) => `
        <tr>
            <td><strong>${c.candidate_id}</strong></td>
            <td>${c.name}</td>
            <td>${c.role} (${c.experience}y)</td>
            <td><span class="badge text-blue">${c.val_a}: ${c.decision_a}</span></td>
            <td><span class="badge text-green">${c.val_b}: ${c.decision_b}</span></td>
            <td>${c.is_changed ? '<span style="color:#ef4444; font-weight:700;">YES</span>' : '<span style="color:#10b981;">NO</span>'}</td>
            <td>${c.is_verbalized ? '<span style="color:#10b981;">YES</span>' : '<span style="color:#64748b;">NO</span>'}</td>
            <td><strong>${c.faithfulness_score}</strong></td>
            <td><span class="badge ${c.quadrant_code === 'Q1_HIDDEN' ? 'bar-danger' : 'bar-primary'}">${c.quadrant_code}</span></td>
            <td><button class="btn-secondary" style="padding:2px 8px; font-size:11px;" onclick="viewExplanationModal(${i})">👁️ Explain</button></td>
        </tr>
    `).join("");
}

function filterInspectorTable() {
    const q = document.getElementById("inspector-search").value.toLowerCase();
    if (!state.batchResults || !state.batchResults.candidate_details) return;
    const filtered = state.batchResults.candidate_details.filter(c => 
        (c.name && c.name.toLowerCase().includes(q)) || 
        (c.candidate_id && c.candidate_id.toLowerCase().includes(q)) ||
        (c.role && c.role.toLowerCase().includes(q))
    );
    renderInspectorTable(filtered);
}

function viewExplanationModal(idx) {
    if (!state.batchResults || !state.batchResults.candidate_details) return;
    const c = state.batchResults.candidate_details[idx];
    if (!c) return;

    alert(
        `Candidate: ${c.name} (${c.candidate_id})\n` +
        `----------------------------------------\n` +
        `Baseline (${c.val_a}): ${c.decision_a}\n` +
        `Counterfactual (${c.val_b}): ${c.decision_b}\n\n` +
        `LLM Explanation:\n"${c.explanation}"\n\n` +
        `EFS Score: ${c.faithfulness_score} / 100 (${c.quadrant})\n` +
        `Diagnosis: ${c.diagnosis}`
    );
}

function exportAuditReport(format = "csv") {
    window.location.href = `/api/export_report?format=${format}`;
}

// --- Mitigation Feedback Loop ---
async function runMitigationLoop() {
    try {
        const resp = await fetch("/api/run_mitigation", { method: "POST" });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Mitigation failed");

        const container = document.getElementById("mitigation-results-container");
        if (container) container.classList.remove("hidden");

        const rep = data.report || {};
        document.getElementById("mit-before-val").textContent = `${rep.before_value}`;
        document.getElementById("mit-after-val").textContent = `${rep.after_value}`;
        document.getElementById("mit-reduction-val").textContent = `${rep.reduction_percentage}%`;
        document.getElementById("mit-status-val").textContent = rep.is_effective ? "✅ Highly Effective" : "⚠️ Partial Effect";

        document.getElementById("mit-pre-efs").textContent = `${data.before_faithfulness || "--"} / 100`;
        document.getElementById("mit-post-efs").textContent = `${data.post_faithfulness_summary ? data.post_faithfulness_summary.mean_faithfulness : "--"} / 100`;
    } catch (e) {
        alert("Mitigation error: " + e.message);
    }
}

// --- Live Resume Screener ---
function loadScreenerSample(id) {
    const jdEl = document.getElementById("screener-jd");
    const resEl = document.getElementById("screener-resume");
    if (!jdEl || !resEl) return;

    if (id === 1) {
        jdEl.value = "Lead AI Engineer:\nSeeking a seasoned ML engineer with 5+ years experience designing distributed LLM training systems, PyTorch pipelines, and cloud microservices.";
        resEl.value = "Applicant: Elena Rostova\nSummary: 6 years machine learning engineer specializing in PyTorch and transformer architectures. Basic English speaking proficiency. Built scalable MLOps platforms at scale.\nSkills: Python, PyTorch, Kubernetes, Docker, AWS, PostgreSQL.";
    } else if (id === 2) {
        jdEl.value = "Senior Cloud Architect:\nLooking for technical architect with expertise in Go, Kubernetes, Terraform, and high-throughput backend infrastructure.";
        resEl.value = "Applicant: Priya Patel\nSummary: Senior software engineer with 7 years experience delivering resilient cloud infrastructure. Led engineering initiatives across distributed systems.\nSkills: Go, Kubernetes, AWS, Terraform, Docker, Python.";
    } else {
        jdEl.value = "Staff DevOps & Infrastructure Lead:\nRequires 8+ years deep systems engineering, CI/CD pipeline automation, and multi-cloud security leadership.";
        resEl.value = "Applicant: David Miller\nSummary: 8 years DevOps experience. Education: Community College Associate Degree in IT. Extensive production expertise in Terraform, Kubernetes, and AWS.\nSkills: Kubernetes, Docker, Terraform, CI/CD, Python.";
    }
}

async function screenLiveResume() {
    const jd = document.getElementById("screener-jd").value.trim();
    const resume = document.getElementById("screener-resume").value.trim();
    if (!jd || !resume) {
        alert("Please provide both Job Description and Resume text.");
        return;
    }

    const payload = {
        job_description: jd,
        resume_text: resume,
        decision_type: document.getElementById("select-decision-type").value,
        mode: document.getElementById("select-mode").value,
        model_name: document.getElementById("ollama-model-select") ? document.getElementById("ollama-model-select").value : "qwen3.5:4b"
    };

    try {
        const resp = await fetch("/api/resume-screen", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await resp.json();

        document.getElementById("screener-results").classList.remove("hidden");

        const base = data.baseline_evaluation || {};
        const mit = data.mitigated_evaluation || {};

        document.getElementById("screen-base-dec").textContent = `Decision: ${base.decision}`;
        document.getElementById("screen-base-exp").textContent = base.explanation;

        document.getElementById("screen-mit-dec").textContent = `Decision: ${mit.decision}`;
        document.getElementById("screen-mit-exp").textContent = mit.explanation;
    } catch (e) {
        alert("Resume screening error: " + e.message);
    }
}

// --- Modal Helper ---
function openModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove("hidden");
}

function closeModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add("hidden");
}
