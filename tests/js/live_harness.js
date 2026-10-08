// A stand-in for the browser, just large enough to run the live dashboard's
// tab page script (live_dashboard._SCRIPT) under a plain JavaScript runtime:
// node (CI) or JavaScriptCore's jsc (macOS). Driven by
// tests/test_live_dashboard.py's _run_live_script, which appends the ids the
// page holds (__IDS), the answers its requests get (__ANSWERS), whether the
// chart library loaded (__NO_PLOTLY), the script itself and a list of steps.
// It checks the script's own logic — what it asks for, writes and draws for
// each answer and click — not a browser's rendering or layout.
//
// getElementById returns null for an id the page does not hold, as a
// browser's does, so a script that reaches a missing element fails the run.
// fetch records each call ({url, cache, headers}) and answers with the next
// of __ANSWERS, a microtask later: {status, body} (body is what json()
// gives; ok is a status in 200-299), {status, bad_json: true} (json()
// fails) or {network_error: true} (the request itself fails). Plotly's
// react, relayout and Plots.resize are recorded, in order, and do nothing.
// ES5 plus Promise, which both runtimes have.

var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __IDS = {};
var __ANSWERS = [];
var __NO_PLOTLY = false;
var __elements = {};
var __fetches = [];
var __plots = [];
var __snaps = {};
var __srcSets = 0;

function __element(id, tag) {
  var el = {id: id, tagName: tag || 'div', style: {}, hidden: false, disabled: false,
            className: '', title: '', children: [], _text: '', _attrs: {},
            _listeners: {}};
  Object.defineProperty(el, 'textContent', {
    get: function() { return el._text; },
    // As a browser's: setting the text replaces the children
    set: function(v) { el._text = String(v); el.children = []; }
  });
  var src = '';
  Object.defineProperty(el, 'src', {
    get: function() { return src; },
    set: function(v) { src = String(v); __srcSets += 1; }
  });
  el.setAttribute = function(name, value) { el._attrs[name] = String(value); };
  el.getAttribute = function(name) { return name in el._attrs ? el._attrs[name] : null; };
  el.appendChild = function(child) { el.children.push(child); return child; };
  el.addEventListener = function(type, fn) {
    (el._listeners[type] = el._listeners[type] || []).push(fn);
  };
  return el;
}

var document = {
  getElementById: function(id) {
    if (!__IDS[id]) { return null; }
    if (!__elements[id]) { __elements[id] = __element(id); }
    return __elements[id];
  },
  createElement: function(tag) { return __element('', tag); }
};

function fetch(url, options) {
  __fetches.push({url: url, cache: options && options.cache,
                  headers: (options && options.headers) || {}});
  var answer = __ANSWERS.shift();
  if (!answer) { return Promise.reject(new Error('no answer queued for ' + url)); }
  if (answer.network_error) { return Promise.reject(new Error('network down')); }
  return Promise.resolve({
    ok: answer.status >= 200 && answer.status < 300,
    status: answer.status,
    json: function() {
      if (answer.bad_json) { return Promise.reject(new Error('not JSON')); }
      return Promise.resolve(JSON.parse(JSON.stringify(answer.body)));
    }
  });
}

function __install() {
  if (__NO_PLOTLY) { return; }
  globalThis.Plotly = {
    react: function(id, data, layout, config) {
      __plots.push({call: 'react', id: id, data: data, layout: layout, config: config});
    },
    relayout: function(id, update) {
      __plots.push({call: 'relayout', id: id, update: update});
    },
    Plots: {resize: function(gd) { __plots.push({call: 'resize', id: gd.id}); }}
  };
}

// Each element the script has touched: its text, class, hidden state,
// tooltip, attributes, frame address and disabled state, and a table body's
// or list's rows (each cell's text, tooltip and left border); and every
// request and chart call since the last snapshot
function __snapshot() {
  var snap = {fetches: __fetches, plots: __plots, srcSets: __srcSets, el: {}};
  __fetches = [];
  __plots = [];
  Object.keys(__elements).forEach(function(id) {
    var el = __elements[id];
    snap.el[id] = {
      text: el._text, className: el.className, hidden: el.hidden, title: el.title,
      attrs: el._attrs, src: el.src, disabled: el.disabled,
      rows: el.children.map(function(child) {
        if (!child.children.length) { return {text: child._text}; }
        return {cells: child.children.map(function(cell) {
          return {text: cell._text, title: cell.title, border: cell.style.borderLeft || ''};
        })};
      })
    };
  });
  return snap;
}

// Let every pending promise run (a bounded number of microtask ticks), then go on
function __spin(ticks, next) {
  var left = ticks;
  (function spin() {
    if (--left <= 0) { next(); } else { Promise.resolve().then(spin); }
  })();
}

// Steps: ["settle"] (let every pending promise run), ["click", id] (a reader
// clicking that element: its click listeners run unless it is disabled),
// ["snap", name]. Emits every snapshot, as JSON, once the steps are done.
function __step(steps, i) {
  if (i >= steps.length) {
    __emit(JSON.stringify(__snaps));
    return;
  }
  var s = steps[i];
  if (s[0] === 'settle') {
    __spin(200, function() { __step(steps, i + 1); });
    return;
  }
  if (s[0] === 'click') {
    var clicked = document.getElementById(s[1]);
    if (!clicked.disabled) {
      (clicked._listeners.click || []).forEach(function(fn) { fn(); });
    }
  } else if (s[0] === 'snap') {
    __snaps[s[1]] = __snapshot();
  }
  __step(steps, i + 1);
}
