"""
HTTP server entrypoint for the Customer Support Agent.
Used for local testing and later for Foundry hosted agent deployment.

Run: python app.py
Test: curl -X POST http://localhost:8087/chat -H "Content-Type: application/json" -d '{"query": "What is the status of order #12345?"}'
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

load_dotenv(override=False)

from agent import run_agent, create_client
from evaluate_foundry import evaluate_single, evaluate_model_only

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Customer Support Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; }
  header { background: #075e54; color: white; padding: 16px 24px; font-size: 18px; font-weight: 600; }
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #dcf8c6; border-bottom-right-radius: 4px; }
  .msg.agent { align-self: flex-start; background: white; border-bottom-left-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
  .msg.agent .tools { margin-top: 8px; padding-top: 8px; border-top: 1px solid #e0e0e0; font-size: 12px; color: #666; }
  .msg.agent .tools summary { cursor: pointer; color: #075e54; font-weight: 500; }
  .typing { align-self: flex-start; background: white; padding: 10px 14px; border-radius: 12px; border-bottom-left-radius: 4px; color: #999; font-style: italic; font-size: 14px; }
  #input-bar { display: flex; gap: 10px; padding: 12px 20px; background: white; border-top: 1px solid #ddd; }
  #query { flex: 1; padding: 10px 14px; border: 1px solid #ccc; border-radius: 20px; font-size: 14px; outline: none; }
  #query:focus { border-color: #075e54; }
  #send { background: #075e54; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; font-size: 18px; cursor: pointer; }
  #send:disabled { background: #ccc; cursor: not-allowed; }
  .suggestions { padding: 10px 20px; display: flex; gap: 8px; flex-wrap: wrap; }
  .suggestions button { background: white; border: 1px solid #075e54; color: #075e54; padding: 6px 12px; border-radius: 16px; font-size: 12px; cursor: pointer; }
  .suggestions button:hover { background: #075e54; color: white; }
  .eval-btn { display: inline-block; margin-top: 8px; padding: 5px 14px; background: #ff9800; color: white; border: none; border-radius: 14px; font-size: 12px; cursor: pointer; font-weight: 500; }
  .eval-btn:hover { background: #e68900; }
  .eval-btn:disabled { background: #ccc; cursor: not-allowed; }
  .eval-result { margin-top: 10px; padding: 12px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; font-size: 13px; }
  .eval-result h4 { margin-bottom: 8px; font-size: 14px; color: #333; }
  .eval-result .pass { color: #28a745; font-weight: 600; }
  .eval-result .fail { color: #dc3545; font-weight: 600; }
  .eval-dim { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .eval-dim .dim-name { width: 130px; font-weight: 500; color: #555; }
  .eval-dim .dim-bar { flex: 1; height: 16px; background: #e9ecef; border-radius: 8px; overflow: hidden; position: relative; }
  .eval-dim .dim-fill { height: 100%; border-radius: 8px; transition: width 0.5s; }
  .eval-dim .dim-score { width: 36px; text-align: right; font-weight: 600; font-size: 12px; }
  .eval-dim .dim-reason { font-size: 11px; color: #777; margin-left: 138px; margin-bottom: 4px; }
  .eval-expected { margin-top: 8px; padding: 8px; background: #e8f5e9; border-radius: 6px; font-size: 11px; color: #2e7d32; }
  .eval-expected b { color: #1b5e20; }
  .detail-section { margin-top: 6px; margin-bottom: 4px; }
  .detail-section summary { cursor: pointer; color: #075e54; font-weight: 500; font-size: 12px; padding: 4px 0; }
  .detail-section summary:hover { text-decoration: underline; }
  .claim-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 4px; }
  .claim-table th { background: #e9ecef; text-align: left; padding: 4px 8px; font-weight: 600; color: #333; border-bottom: 2px solid #dee2e6; }
  .claim-table td { padding: 4px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
  .claim-table .grounded { color: #28a745; font-weight: 600; }
  .claim-table .ungrounded { color: #dc3545; font-weight: 600; }
  .checklist { list-style: none; padding: 0; margin: 4px 0; font-size: 11px; }
  .checklist li { padding: 2px 0; }
  .checklist .check { color: #28a745; }
  .checklist .cross { color: #dc3545; }
  .model-eval-btn { display: inline-block; margin-top: 8px; margin-left: 6px; padding: 5px 14px; background: #2196f3; color: white; border: none; border-radius: 14px; font-size: 12px; cursor: pointer; font-weight: 500; }
  .model-eval-btn:hover { background: #1976d2; }
  .model-eval-btn:disabled { background: #ccc; cursor: not-allowed; }
  .model-eval-result { border-left: 3px solid #2196f3; }
  .model-response-box { margin: 8px 0; padding: 8px; background: #fff3e0; border-radius: 6px; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; }
</style>
</head>
<body>
<header>Customer Support Agent</header>
<div id="messages">
  <div class="msg agent">Hello! I'm your customer support assistant. How can I help you today?</div>
</div>
<div class="suggestions">
  <button onclick="ask(this.textContent)">I ordered a laptop 2 weeks ago (order #12345) and it still hasn't arrived. Can you check the status and tell me what my options are?</button>
  <button onclick="ask(this.textContent)">I want a refund for order #67890, the product was damaged when it arrived.</button>
  <button onclick="ask(this.textContent)">Can I cancel order #11111? I changed my mind.</button>
  <button onclick="ask(this.textContent)">What's the status of order #99999?</button>
  <button onclick="ask(this.textContent)">I want to return my wireless mouse from order #67890. How do I do that?</button>
  <button onclick="ask(this.textContent)">How fast can I get a new order delivered? What are the shipping options?</button>
  <button onclick="ask(this.textContent)">I never received order #12345 and I want my money back immediately. This is unacceptable!</button>
  <button onclick="ask(this.textContent)">Tell me about order #00000</button>
</div>
<div id="input-bar">
  <input id="query" type="text" placeholder="Type your message..." autocomplete="off">
  <button id="send" onclick="send()">&#9654;</button>
</div>
<script>
  const msgs = document.getElementById('messages');
  const input = document.getElementById('query');
  const btn = document.getElementById('send');

  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !btn.disabled) send(); });

  function ask(text) { input.value = text; send(); }

  function barColor(score, max) {
    const pct = score / max;
    if (pct >= 0.8) return '#28a745';
    if (pct >= 0.6) return '#ff9800';
    return '#dc3545';
  }

  function renderEvalResult(container, data) {
    const dims = data.scores || {};
    const reasons = data.reasons || {};
    const pass = data.pass;
    const avg = data.average_score;
    const expected = data.expected_behavior;
    const analysis = data.detailed_analysis || {};

    let html = '<div class="eval-result">';
    html += '<h4>Foundry Evaluation Result &mdash; <span class="' + (pass ? 'pass' : 'fail') + '">' + (pass ? 'PASS' : 'FAIL') + '</span> (avg ' + avg + '/5)</h4>';

    const dimOrder = ['relevance','coherence','groundedness','fluency','task_adherence','intent_resolution','tool_call_accuracy','response_completeness'];
    const dimLabels = {
      relevance: 'Relevance', coherence: 'Coherence', groundedness: 'Groundedness',
      fluency: 'Fluency', task_adherence: 'Task Adherence', intent_resolution: 'Intent Resolution',
      tool_call_accuracy: 'Tool Call Accuracy', response_completeness: 'Resp. Completeness'
    };

    for (const dim of dimOrder) {
      if (!(dim in dims)) continue;
      const score = dims[dim];
      const isBinary = (dim === 'task_adherence');
      const max = isBinary ? 1 : 5;
      const pct = (score / max) * 100;
      const label = isBinary ? (score >= 1 ? '1 (Yes)' : '0 (No)') : score + '/5';
      const color = barColor(score, max);
      const name = dimLabels[dim] || dim;

      html += '<div class="eval-dim">';
      html += '  <span class="dim-name">' + name + '</span>';
      html += '  <div class="dim-bar"><div class="dim-fill" style="width:' + pct + '%;background:' + color + '"></div></div>';
      html += '  <span class="dim-score" style="color:' + color + '">' + label + '</span>';
      html += '</div>';
      if (reasons[dim]) {
        html += '<div class="eval-dim"><span class="dim-reason">' + reasons[dim] + '</span></div>';
      }

      // Detailed analysis per dimension
      const da = analysis[dim];
      if (da) {
        html += '<details class="detail-section"><summary>View detailed analysis</summary>';

        if (dim === 'groundedness' && da.claims && da.claims.length > 0) {
          html += '<table class="claim-table"><tr><th>Claim</th><th>Evidence</th><th>Grounded?</th></tr>';
          da.claims.forEach(c => {
            const cls = c.grounded ? 'grounded' : 'ungrounded';
            const icon = c.grounded ? '&#10003;' : '&#10007;';
            html += '<tr><td>' + esc(c.claim) + '</td><td>' + esc(c.evidence) + '</td><td class="' + cls + '">' + icon + '</td></tr>';
          });
          html += '</table>';
        }

        else if (dim === 'relevance' && (da.addressed_aspects || da.missed_aspects)) {
          html += '<ul class="checklist">';
          (da.addressed_aspects || []).forEach(a => { html += '<li><span class="check">&#10003;</span> ' + esc(a) + '</li>'; });
          (da.missed_aspects || []).forEach(a => { html += '<li><span class="cross">&#10007;</span> Missed: ' + esc(a) + '</li>'; });
          html += '</ul>';
        }

        else if (dim === 'task_adherence' && (da.rules_followed || da.rules_violated)) {
          html += '<ul class="checklist">';
          (da.rules_followed || []).forEach(r => { html += '<li><span class="check">&#10003;</span> ' + esc(r) + '</li>'; });
          (da.rules_violated || []).forEach(r => { html += '<li><span class="cross">&#10007;</span> ' + esc(r) + '</li>'; });
          html += '</ul>';
        }

        else if (dim === 'intent_resolution' && (da.resolved || da.unresolved)) {
          html += '<ul class="checklist">';
          (da.resolved || []).forEach(r => { html += '<li><span class="check">&#10003;</span> ' + esc(r) + '</li>'; });
          (da.unresolved || []).forEach(r => { html += '<li><span class="cross">&#10007;</span> Unresolved: ' + esc(r) + '</li>'; });
          html += '</ul>';
        }

        else if (da.strengths || da.weaknesses) {
          html += '<ul class="checklist">';
          (da.strengths || []).forEach(s => { html += '<li><span class="check">&#10003;</span> ' + esc(s) + '</li>'; });
          (da.weaknesses || []).forEach(w => { html += '<li><span class="cross">&#10007;</span> ' + esc(w) + '</li>'; });
          html += '</ul>';
        }

        html += '</details>';
      }
    }

    if (expected) {
      html += '<div class="eval-expected"><b>Expected behavior:</b> ' + expected + '</div>';
    }
    html += '</div>';
    container.innerHTML += html;
  }

  function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

  async function runEval(query, evalBtn, container) {
    evalBtn.disabled = true;
    evalBtn.textContent = 'Evaluating...';

    try {
      const res = await fetch('/evaluate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query})
      });
      const data = await res.json();
      evalBtn.style.display = 'none';
      if (data.error) {
        container.innerHTML += '<div class="eval-result" style="color:#dc3545">Evaluation error: ' + data.error + '</div>';
      } else {
        renderEvalResult(container, data);
      }
    } catch(err) {
      evalBtn.disabled = false;
      evalBtn.textContent = 'Evaluate Agent';
      container.innerHTML += '<div class="eval-result" style="color:#dc3545">Error: ' + err.message + '</div>';
    }
    msgs.scrollTop = msgs.scrollHeight;
  }

  async function runModelEval(query, evalBtn, container) {
    evalBtn.disabled = true;
    evalBtn.textContent = 'Running model...';

    try {
      const res = await fetch('/evaluate-model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query})
      });
      const data = await res.json();
      evalBtn.style.display = 'none';
      if (data.error) {
        container.innerHTML += '<div class="eval-result model-eval-result" style="color:#dc3545">Model eval error: ' + data.error + '</div>';
      } else {
        renderModelEvalResult(container, data);
      }
    } catch(err) {
      evalBtn.disabled = false;
      evalBtn.textContent = 'Compare Model (No Tools)';
      container.innerHTML += '<div class="eval-result model-eval-result" style="color:#dc3545">Error: ' + err.message + '</div>';
    }
    msgs.scrollTop = msgs.scrollHeight;
  }

  function renderModelEvalResult(container, data) {
    const scores = data.scores || {};
    const reasons = data.reasons || {};
    const pass = data.pass;
    const avg = data.average_score;
    const modelResponse = data.model_response || '';
    const expected = data.expected_behavior;

    let html = '<div class="eval-result model-eval-result">';
    html += '<h4>Model Evaluation (No Tools) &mdash; <span class="' + (pass ? 'pass' : 'fail') + '">' + (pass ? 'PASS' : 'FAIL') + '</span> (avg ' + avg + '/5)</h4>';

    html += '<div class="model-response-box"><b>Model response (without tools):</b><br>' + esc(modelResponse) + '</div>';

    const dimOrder = ['relevance','coherence','groundedness','fluency','intent_resolution','response_completeness'];
    const dimLabels = {
      relevance: 'Relevance', coherence: 'Coherence', groundedness: 'Groundedness',
      fluency: 'Fluency', intent_resolution: 'Intent Resolution',
      response_completeness: 'Resp. Completeness'
    };

    for (const dim of dimOrder) {
      if (!(dim in scores)) continue;
      const score = scores[dim];
      const max = 5;
      const pct = (score / max) * 100;
      const label = score + '/5';
      const color = barColor(score, max);
      const name = dimLabels[dim] || dim;

      html += '<div class="eval-dim">';
      html += '  <span class="dim-name">' + name + '</span>';
      html += '  <div class="dim-bar"><div class="dim-fill" style="width:' + pct + '%;background:' + color + '"></div></div>';
      html += '  <span class="dim-score" style="color:' + color + '">' + label + '</span>';
      html += '</div>';
      if (reasons[dim]) {
        html += '<div class="eval-dim"><span class="dim-reason">' + esc(reasons[dim]) + '</span></div>';
      }
    }

    if (expected) {
      html += '<div class="eval-expected"><b>Expected behavior:</b> ' + esc(expected) + '</div>';
    }
    html += '</div>';
    container.innerHTML += html;
  }

  async function send() {
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    btn.disabled = true;

    const userDiv = document.createElement('div');
    userDiv.className = 'msg user';
    userDiv.textContent = q;
    msgs.appendChild(userDiv);

    const typing = document.createElement('div');
    typing.className = 'typing';
    typing.textContent = 'Thinking...';
    msgs.appendChild(typing);
    msgs.scrollTop = msgs.scrollHeight;

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: q})
      });
      const data = await res.json();
      typing.remove();

      const agentDiv = document.createElement('div');
      agentDiv.className = 'msg agent';

      let html = data.response || data.error || 'No response';
      if (data.tool_calls && data.tool_calls.length > 0) {
        html += '<div class="tools"><details><summary>Tools used (' + data.tool_calls.length + ')</summary>';
        data.tool_calls.forEach(tc => {
          html += '<div style="margin-top:4px"><b>' + tc.tool + '</b>(' + JSON.stringify(tc.arguments) + ')</div>';
        });
        html += '</details></div>';
      }

      // Evaluate buttons
      const evalId = 'eval-' + Date.now();
      const qEsc = q.replace(/\\\\/g,'\\\\\\\\').replace(/'/g,"\\\\'")
      html += '<button class="eval-btn" id="' + evalId + '-btn" onclick="runEval(\\''+qEsc+'\\',' +
              'document.getElementById(\\'' + evalId + '-btn\\'),' +
              'document.getElementById(\\'' + evalId + '-result\\'))">' + 'Evaluate Agent</button>';
      html += '<button class="model-eval-btn" id="' + evalId + '-model-btn" onclick="runModelEval(\\''+qEsc+'\\',' +
              'document.getElementById(\\'' + evalId + '-model-btn\\'),' +
              'document.getElementById(\\'' + evalId + '-model-result\\'))">' + 'Compare Model (No Tools)</button>';
      html += '<div id="' + evalId + '-result"></div>';
      html += '<div id="' + evalId + '-model-result"></div>';

      agentDiv.innerHTML = html;
      msgs.appendChild(agentDiv);
    } catch(err) {
      typing.remove();
      const errDiv = document.createElement('div');
      errDiv.className = 'msg agent';
      errDiv.textContent = 'Error: ' + err.message;
      msgs.appendChild(errDiv);
    }
    btn.disabled = false;
    msgs.scrollTop = msgs.scrollHeight;
    input.focus();
  }
</script>
</body>
</html>"""


class AgentHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for the customer support agent."""

    client = None
    # Store the last agent result for each query so /evaluate can use it
    last_results = {}

    def do_POST(self):
        if self.path == "/chat":
            self._handle_chat()
        elif self.path == "/evaluate":
            self._handle_evaluate()
        elif self.path == "/evaluate-model":
            self._handle_evaluate_model()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_chat(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 1_000_000:
            self._send_json(413, {"error": "Request too large"})
            return
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        query = data.get("query", "")
        if not query:
            self._send_json(400, {"error": "Missing 'query' field"})
            return

        if AgentHandler.client is None:
            AgentHandler.client = create_client()

        result = run_agent(query, client=AgentHandler.client)

        # Store full result for later evaluation
        AgentHandler.last_results[query] = result

        self._send_json(200, {
            "response": result["response"],
            "tool_calls": result["tool_calls"],
        })

    def _handle_evaluate(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 1_000_000:
            self._send_json(413, {"error": "Request too large"})
            return
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        query = data.get("query", "")
        if not query:
            self._send_json(400, {"error": "Missing 'query' field"})
            return

        # Look up the stored agent result
        stored = AgentHandler.last_results.get(query)
        if not stored:
            self._send_json(400, {"error": "No agent response found for this query. Send the query via chat first."})
            return

        try:
            eval_result = evaluate_single(
                query=query,
                response=stored["response"],
                tool_calls=stored["tool_calls"],
                messages=stored["messages"],
            )
            self._send_json(200, eval_result)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_evaluate_model(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 1_000_000:
            self._send_json(413, {"error": "Request too large"})
            return
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        query = data.get("query", "")
        if not query:
            self._send_json(400, {"error": "Missing 'query' field"})
            return

        try:
            eval_result = evaluate_model_only(query=query)
            self._send_json(200, eval_result)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(CHAT_HTML.encode())
        elif self.path == "/health":
            self._send_json(200, {"status": "healthy"})
        else:
            self._send_json(404, {"error": "Not found"})

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        print(f"  [{self.command}] {self.path} - {args[0] if args else ''}")


def main():
    port = int(os.environ.get("PORT", "8087"))
    server = HTTPServer(("0.0.0.0", port), AgentHandler)
    print(f"Customer Support Agent server running on http://localhost:{port}")
    print(f"  POST /chat  — Send a query to the agent")
    print(f"  GET  /health — Health check")
    print(f"Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
