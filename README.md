# TurboSim in the browser

TurboSim, a thermodynamic cycle simulator, running entirely in your web browser: open https://colintort.github.io/turbosim/ and start drawing. The Python solver (CoolProp property data, NumPy, SciPy) runs inside the page with Pyodide, so nothing is installed and nothing you draw is sent anywhere. The first visit downloads about 40 MB, which the browser keeps for next time.

`checks.html` runs the 42 textbook and ME 313 checks in your browser.

These files are built by `npm run build:web` in TurboSim's `frontend` folder. Edit the source there, not here.
