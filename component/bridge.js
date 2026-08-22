(function () {
  var READY = false;
  var rendered = null;
  var dismissed = false;

  function post(msg) {
    window.parent.postMessage(Object.assign({isStreamlitMessage: true}, msg), "*");
  }
  function setHeight(h) {
    post({type: "streamlit:setFrameHeight", height: Math.ceil(h)});
  }
  function sendValue(value) {
    post({type: "streamlit:setComponentValue", value: value, dataType: "json"});
  }

  function hideServerOnlyBits() {
    ['secJobs'].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
    document.querySelectorAll('.rail, .mini-header').forEach(function (el) {
      el.style.display = 'none';
    });
  }

  function measure() {
    var report = document.getElementById('reportView');
    var showing = report && report.classList.contains('show');
    setHeight(showing
      ? Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
      : 780);
  }

  function toBase64(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () { resolve(r.result.split(',')[1]); };
      r.onerror = reject;
      r.readAsDataURL(file);
    });
  }

  window.analyze = async function (opts) {
    opts = opts || {};
    if (!currentFile) return;
    document.getElementById('uploadActions').classList.remove('show');
    document.getElementById('errBox').classList.remove('show');
    startLog(performance.now());
    sendValue({
      action: 'analyze',
      filename: currentFile.name,
      data: await toBase64(currentFile),
      target_soc: opts.targetSoc || '',
      job_description: opts.jobDescription || '',
      stage: (window.selectedStage || '')
    });
  };

  var _reset = window.resetView;
  window.resetView = function () {
    dismissed = true;
    if (_reset) _reset.apply(this, arguments);
    sendValue({action: 'reset'});
    setTimeout(measure, 60);
  };

  window.addEventListener("message", function (e) {
    var d = e.data || {};
    if (d.type !== "streamlit:render") return;
    var args = d.args || {};
    var key = args.report ? JSON.stringify(args.report).length + ':' + (args.report.resume || '') : null;
    if (args.report && key !== rendered && !dismissed) {
      rendered = key;
      clearLog();
      renderReport(args.report, args.elapsed || null);
      hideServerOnlyBits();
    }
    if (!args.report) { rendered = null; dismissed = false; }
    if (typeof args.count === 'number' && window.setAnalysisCount) {
      setAnalysisCount(args.count);
    }
    requestAnimationFrame(function () { requestAnimationFrame(measure); });
    setTimeout(measure, 700);
    setTimeout(measure, 1800);
  });

  window.addEventListener('resize', measure);
  new MutationObserver(measure).observe(document.body,
    {childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style']});

  if (!READY) {
    READY = true;
    post({type: "streamlit:componentReady", apiVersion: 1});
    measure();
  }
})();
