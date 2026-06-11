// for connecting the frontend to the fastapi engine

document.addEventListener("DOMContentLoaded", async () => {
    const userInput = document.getElementById('user-input');
    const sendBTN = document.getElementById('send-btn');
    const outputLog = document.getElementById('output-log');

    // api connecting config
    // Automatically detect the environment
    const isLocal = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";

    // Route to localhost during dev, and Render during production
    const API_BASE_URL = isLocal 
        ? "http://127.0.0.1:7860/api/v1" 
        : "https://eericzzhao-fnv-rag-backend.hf.space/api/v1";     

    // dom-aware typewriter/terminal effect
    async function typewriterHTML(targetElement, htmlString, speed = 15) {
        //create a hidden temp container to parse the HTML
        const tempDiv = document.createElement('div')
        tempDiv.innerHTML = htmlString;
        targetElement.innerHTML = '';

        // recursive func to handle elements instantly but delay text
        async function processNode(node, parent) {
            if (node.nodeType === Node.TEXT_NODE) {
                const text = node.textContent;
                for (let i = 0; i < text.length; i++){
                    parent.appendChild(document.createTextNode(text[i]));
                    //keeps the terminal autoscrolling as the text populates
                    outputLog.scrollTop = outputLog.scrollHeight;
                    // delay next chracater
                    await new Promise(r => setTimeout(r, speed));
                }
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                // clone the HTML tag instnatly but empty
                const clone = node.cloneNode(false);
                parent.appendChild(clone);
                // recusrively type the text inside the tag
                for (const child of node.childNodes) {
                    await processNode(child, clone);
                }
            }
        }

        // start the typing cool thingy
        for (const child of tempDiv.childNodes) {
            await processNode(child, targetElement);
        }
    }

    async function appendMessage(text, isUser = false, isMarkdown = false, animate = false) {
        const msgDiv = document.createElement('div'); //div handles block elemetns into lists
        if (isUser) {
            msgDiv.textContent = `> ${text}`;
            msgDiv.style.color = `#fff`;
            outputLog.appendChild(msgDiv);
        } else { 
            msgDiv.style.color = 'var(--robco-green)';
            outputLog.appendChild(msgDiv); // append empty div to screen first

            if (animate) {
                // parse markdown THEn type it out
                const htmlContent = isMarkdown ? marked.parse(text) : text;
                await typewriterHTML(msgDiv, htmlContent, 18); //10 ms delay
            } else {
                // instantly print (loading messages)
                if (isMarkdown) {
                    msgDiv.innerHTML = marked.parse(text);
                } else {
                    msgDiv.textContent = text;
                }
            }
        }
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

                const safeScore = (contextData && contextData.rerank_score !== undefined)
                    ? contextData.rerank_score
                    : 0.0;
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
        // appendMessage("PROCESSING QUERY...", false);
        let loadingInterval;
        const loadingDiv = document.createElement('div');
        loadingDiv.style.color = 'var(--robco-green)';

        function startLoadingAnimations() {
            outputLog.appendChild(loadingDiv);
            const steps = [
                "> Initializing terminal link...",
                "> Vectorizing user query...",
                "> Scanning load order matrices...",
                "> Retrieving Nexus Mods telemetry...",
                "> Applying cross-encorder re-ranking...",
                "> Synthesizing diagnostic report..."
            ];
            let stepIndex = 0;
            loadingDiv.innerText = steps[stepIndex];
            outputLog.scrollTop = outputLog.scrollHeight;

            loadingInterval = setInterval(() => {
                stepIndex++;
                if (stepIndex < steps.length) {
                    loadingDiv.innerText = steps[stepIndex];
                } else {
                    loadingDiv.innerText = steps[steps.length - 1] + " _";               
                }
                outputLog.scrollTop = outputLog.scrollHeight;
            }, 1200); // change step every 1.2 seconds

            const circles = d3.selectAll("circle");
            if (!circles.empty()) {
                function pulse() {
                    circles.transition()
                        .duration(600)
                        .attr("r", d => d.radius ? d.radius + 4 : 10) // expand
                        .style("opacity", 0.4)
                        .transition()
                        .duration(600)
                        .attr("r", d => d.radius ? d.radius : 6) // Shrink back to normal
                        .style("opacity", 1)
                        .on("end", pulse);

                }
                pulse();
            }
        }
        function stopLoadingAnimations() {
            clearInterval(loadingInterval);
            if (outputLog.contains(loadingDiv)) {
                outputLog.removeChild(loadingDiv);
            }
            d3.selectAll("circle").interrupt() // stops the pulsing
        }

        startLoadingAnimations();

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

            stopLoadingAnimations();
            // removes the "processing..."" text
            //outputLog.removeChild(outputLog.lastChild);

            // print the header immediately
            appendMessage(`> DATA RETRIEVED:`, false, false, true);

            await appendMessage(data.answer, false, true, true);
            renderRAGGraph(data)

        } catch (error) {
            outputLog.removeChild(outputLog.lastChild);
            appendMessage(`> ERROR: CONNECTION TO SERVER FAILED ${error.message}`);
        }
    }

    const chatForm = document.getElementById('chat-form');
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault(); //no page refrsh
        handleQuery();
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
    await appendMessage("System initialized. Knowledge base loaded.", false, false, true);
    await appendMessage("Awaitng user input...", false, false, true); 

})