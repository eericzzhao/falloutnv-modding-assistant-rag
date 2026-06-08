// for connecting the frontend to the fastapi engine

document.addEventListener("DOMContentLoaded", () => {
    const userInput = document.getElementById('user-input');
    const sendBTN = document.getElementById('send-btn');
    const outputLog = document.getElementById('output-log');

    // api connecting config
    const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

    function appendMessage(text, isUser = false) {
        const msgDiv = document.createElement('p');
        msgDiv.textContent = isUser ? `> ${text}` : text;
        msgDiv.style.color = isUser ?  `#fff` : 'var(--robco-green)';
        outputLog.appendChild(msgDiv);
        // auto-scroll
        outputLog.scrollTop = outputLog.scrollHeight; 
    }

    async function handleQuery() {
        function renderRAGGraph(payload) {
            const svg = d3.select("#rag-visualizer");
            svg.selectAll('*').remove(); // clears the previosu graph

            const width = document.getElementById("d3-container").clientWidth;
            const height = document.getElementById("d3-container").clientHeight;

            // 1.Data Prep: mapping candidates to nodes
            const selectedMap = new Map(payload.selected_context.map(c => [c.text, c]));

            const nodes = payload.candidates.map(candidate => {
                const isSelected = selectedMap.has(candidate.text);
                const contextData = selectedMap.get(candidate.text);

                // scale radius based on the huggingface rerank score
                const radius = isSelected ? 8 + (contextData.rerank_score * 12) : 5;

                return {
                    id: candidate.text,
                    isSelected: isSelected,
                    radius: radius,
                    color: isSelected ? "var(--robco-green)" : "#1a401a",
                    score: isSelected ? contextData.rerank_score.toFixed(3) : "Discarded", 
                    source: candidate.source_file
                };
            });

            // adds a central "Query" node
            nodes.push({ id: "QUERY", isSelected: true, radius: 20, color: "#fff", source: "User Input", score: "N/A" });

            // 1. link all selected nodes to the centry query
            const links = payload.selected_context.map(c => ({
                source: c.text,
                target: "QUERY"
            }));

            // 2. setup d3 force simulation
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(50))
                .force("charge", d3.forceManyBody().strength(-30))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(d => d.radius + 2));

            // 3. draw elements
            const tooltip = d3.select("#d3-tooltip");
            const link = svg.append("g")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("stroke", "var(--robco-green)")
                .attr("stroke-opacity", 0.6);

            const node = svg.append("g")
                    .selectAll("circle")
                    .data(nodes)
                    .join("circle")
                    .attr("r", d => d.radius)
                    .attr("fill", d => d.color)
                    .on("mouseover", function(event, d) {
                        d3.select(this).attr("stroke", "#fff").attr("stroke-width", 2);
    
                    tooltip.transition().duration(200).style("opacity", .95);
                        tooltip.html(`<strong>File:</strong> ${d.source}<br/><strong>Match Score:</strong> ${d.score}<br/><hr style="border-color:var(--robco-green);"/>${d.id.substring(0, 80)}...`)
                            .style("left", (event.pageX + 15) + "px")
                            .style("top", (event.pageY - 28) + "px");
                    })
                    .on("mouseout", function(event, d) {
                        d3.select(this).attr("stroke", null);
                        tooltip.transition().duration(500).style("opacity", 0);
                    })
                    .call(d3.drag() 
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
            
            // 4. tick function to update positions
            simulation.on("tick", () => {
                link.attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y)

                node.attr("cx", d => d.x)
                    .attr("cy", d => d.y);
            });

            // drag simulation functions
            function dragstarted(event){
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            
        }
        const query = userInput.value.trim();
        if (!query) return;
        
        appendMessage(query, true);
        userInput.value = '';
        appendMessage("PROCESSING QUERY...", false);

        try {
            const response = await fetch( `${API_BASE_URL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json'},
                body: JSON.stringify({ question: query})
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                // forces a print of the bug
                throw new Error(errorData.detail || `HTTP Error ${response.status}`);
            }
            const data = await response.json();
            // removes the "processing..."" text
            outputLog.removeChild(outputLog.lastChild);

            appendMessage(`> DATA RETRIEVED: ${data.answer}`);

            renderRAGGraph(data)
            console.log("Telemetry Payload for D3.js:", data);
        } catch (error) {
            outputLog.removeChild(outputLog.lastChild);
            appendMessage(`> ERROR: CONNECTION TO SERVER FAILED ${error.message}`);
        }
    }

    sendBTN.addEventListener('click', (e) => {
        e.preventDefault();
        handleQuery();
    });
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault(); 
            handleQuery();
        }
    });

    // drag and drop logic for the loadorder txt files
    const dropZone = document.getElementById('drop-zone');

    dropZone.addEventListener('click', () =>
    document.getElementById('file-input').click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = 'var(--robco-dark)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.backgroundColor = 'transparent';
    });

    dropZone.addEventListener('drop', async(e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = 'transparent';

        if (e.dataTransfer.files.length) {
            const file = e.dataTransfer.files[0];
            appendMessage(`> UPLOADING FILE: ${file.name}...`);

            const formData = new FormData();
            formData.append("file", file);

            try {
                const response = await fetch(`${API_BASE_URL}/analyze-load-order`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                appendMessage(`> DIAGNOSTICS COMPLETE> ISSUES FOUND ${data.issues_detected}`);
                data.diagnostics.forEach(diag => {
                    appendMessage(`> WARNING [${diag.mod_name}]: ${diag.issue_description}`);
                });
            } catch (error) {
                outputLog.removeChild(outputLog.lastChild);
                appendMessage(`> ERROR: DIAGNOSTIC FAILED. ${error.message}`);
            }
        }
    });

})