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
        const query = userInput.ariaValueMax.trim();
        if (!query) return;
        
        appendMesssage(query, true);
        userInput.value = '';
        appendMessage("PROCESSING QUERY...", false);

        try {
            const response = await fetch( `${API_BASE_URL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json'},
                body: JSON.stringify({ question: query})
            });
            
            if (!response.ok) throw new Error("Newwork response was not ok");

            const data = await response.json();
            // removes the "processing..."" text
            outputLog.removeChild(outputLog.lastChild);

            appendMessage(`> DATA RETRIEVED: ${data.answer}`);

            // console log the candidates and scores + d3 visualizaiton
            console.log("Telemetry Payload for D3.js:", data);
        } catch (error) {
            outputLog.removeChild(outputLog.lastChild);
            appendMessage(`> ERROR: CONNECTION TO SERVER FAILED ${error.message}`);
        }
    }

    sendBTN.addEventListener('click', handleQuery);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleQuery();
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
                    appendMesssage(`> WARNING [${diag.mod_name}]: ${diag.issue_description}`);
                });
            } catch (error) {
                outputLog.removeChild(outputLog.lastChild);
                appendMessage(`> ERROR: DIAGNOSTIC FAILED. ${error.message}`);
            }
        }
    });

})