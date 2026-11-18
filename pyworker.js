/* TurboSim in the browser: the solver.
 *
 * A Web Worker that runs TurboSim's own Python backend (backend/*.py, copied
 * in by `npm run build:web`) on Pyodide, so the page never freezes while it
 * solves. The page posts { id, route, body } and gets back { id, json } (the
 * same JSON the desktop app's /<route> returns), { id, pdf } for a report, or
 * { id, error }. Requests run one at a time, in order.
 *
 * Pyodide 0.29.2 or later is needed: older releases ship CoolProp 6.6, and
 * TurboSim (and its textbook checks) use CoolProp 7.2.
 */
/* eslint-disable no-restricted-globals */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
const REPORTLAB = "reportlab==5.1.0"; // the version the .exe is built with
const HOME = "/home/pyodide/backend";
const V = new URL(self.location.href).searchParams.get("v") || "";
const q = (path) => (V ? `${path}?v=${encodeURIComponent(V)}` : path);

const status = (state, text) => self.postMessage({ type: "status", state, text });

const DISPATCH = `
import json, math, sys
sys.path.insert(0, ${JSON.stringify(HOME)})
import routes


def _default(o):
    if hasattr(o, "tolist"):        # numpy numbers and arrays
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON")


def _finite(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if hasattr(o, "tolist"):
        return _finite(o.tolist())
    return o


def web_call(route, text):
    out = routes.JSON_ROUTES[route](json.loads(text))
    try:
        return json.dumps(out, default=_default, allow_nan=False)
    except ValueError:              # NaN or infinity somewhere: send null, as JSON can't hold them
        return json.dumps(_finite(out), default=_default)


def web_report(text):
    from report import ReportError, build_report
    try:
        return build_report(json.loads(text))
    except ReportError as e:
        return str(e)


def web_validate():
    import io, contextlib, validate_textbook as vt
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = vt.print_table(vt.run())
    return json.dumps({"ok": bool(ok), "text": buf.getvalue()})
`;

let py = null;
let reportReady = null;

async function start() {
  status("loading", "Starting Python");
  importScripts(`${PYODIDE}pyodide.js`);
  py = await self.loadPyodide({ indexURL: PYODIDE });
  status("loading", "Loading CoolProp, NumPy and SciPy");
  await py.loadPackage(["numpy", "scipy", "coolprop"]);
  status("loading", "Loading the TurboSim solver");
  const files = await (await fetch(q("files.json"))).json();
  py.FS.mkdirTree(HOME);
  await Promise.all(
    files.map(async (f) => {
      const res = await fetch(q(`backend/${f}`));
      if (!res.ok) throw new Error(`backend/${f} could not be loaded (${res.status})`);
      py.FS.writeFile(`${HOME}/${f}`, await res.text());
    })
  );
  py.runPython(DISPATCH);
  // warm up: the first solve imports the solver and CoolProp's fluids
  py.runPython("import system_solver, workings, exergy, propulsion");
  status("ready", "Ready");
}

// PDF reports need ReportLab (from PyPI) and the report fonts; fetched the
// first time a report is made, not on every visit.
function loadReportTools() {
  if (!reportReady) {
    reportReady = (async () => {
      await py.loadPackage(["micropip"]);
      await py.runPythonAsync(`import micropip\nawait micropip.install(${JSON.stringify(REPORTLAB)})`);
      py.FS.mkdirTree(`${HOME}/fonts`);
      for (const f of ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]) {
        const res = await fetch(q(`backend/fonts/${f}`));
        if (res.ok) py.FS.writeFile(`${HOME}/fonts/${f}`, new Uint8Array(await res.arrayBuffer()));
      }
    })().catch((e) => {
      reportReady = null; // try again next time
      throw e;
    });
  }
  return reportReady;
}

// Python's own error message, without the traceback
function pyMessage(e) {
  const lines = String(e && e.message ? e.message : e).trim().split("\n");
  return lines[lines.length - 1] || "The solver hit an error.";
}

async function handle({ id, route, body }) {
  try {
    await ready;
    if (route === "report") {
      await loadReportTools();
      const out = py.globals.get("web_report")(body);
      if (typeof out === "string") return self.postMessage({ id, error: out });
      const pdf = out.toJs();
      out.destroy();
      return self.postMessage({ id, pdf }, [pdf.buffer]);
    }
    if (route === "validate") {
      return self.postMessage({ id, json: py.globals.get("web_validate")() });
    }
    const json = py.globals.get("web_call")(route, body);
    self.postMessage({ id, json });
  } catch (e) {
    self.postMessage({ id, error: pyMessage(e) });
  }
}

const ready = start().catch((e) => {
  status("failed", pyMessage(e));
  throw e;
});

let queue = Promise.resolve();
self.onmessage = (e) => {
  queue = queue.then(() => handle(e.data));
};
