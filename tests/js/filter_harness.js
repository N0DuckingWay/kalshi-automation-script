// A stand-in for the browser, just large enough to run the dashboard's
// page-wide filter script (dashboard._FILTER_JS) — and, when a test asks for
// it and the page carries it, the scenario explorer's script
// (dashboard._SCENARIO_EXPLORER_JS) before it, as the page orders them —
// under a plain JavaScript runtime: node (CI) or JavaScriptCore's jsc
// (macOS). Driven by tests/test_dashboard.py's _run_script, which appends the
// page's elements, every packed data block of the page (already inflated:
// each script's own inflate / unpack is replaced by __inflate below), the
// scripts themselves and a list of steps. It checks the scripts' own logic —
// what they draw, write, load and enable for each choice — not a browser's
// rendering or layout (Plotly.Plots.resize does nothing here). ES5 plus
// Promise and Object.assign, which both runtimes have.
//
// Strict mode (page.strict): getElementById returns null for an id the page
// does not hold, as a browser's does, so a script that dereferences a missing
// element fails the run. Otherwise any id asked for is created on the fly.
//
// Deferred blocks (__DEFERRED): their inflate waits for a ["resolve", id] or
// ["reject", id] step, as a slow DecompressionStream's would, so every order
// in which choices and arriving chunks can interleave can be driven on
// purpose. Every other block inflates at once.

var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __elements = {};
var __reacts = [];
var __updates = [];
var __snaps = {};
var __STRICT = false;
var __IDS = {};
// Every packed block of the page, inflated, by element id; the ids in
// __DAMAGED inflate with an error, as a damaged block's would
var __BLOCKS = {};
var __DAMAGED = {};
// The ids whose inflate waits for a step, and the inflates waiting, by id
var __DEFERRED = {};
var __PENDING = {};
// The element ids the script inflated, in order
var __inflated = [];

function __escape(text) {
  return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function __element(id, tag) {
  var el = {id: id, tagName: tag || 'div', style: {}, innerHTML: '', disabled: false,
            parentElement: {style: {}}, data: null, layout: null, children: [],
            _text: '', _listeners: {}};
  // As a browser's: setting textContent replaces the children and makes the
  // markup the text, escaped
  Object.defineProperty(el, 'textContent', {
    get: function() { return el._text; },
    set: function(v) { el._text = String(v); el.children = []; el.innerHTML = __escape(v); }
  });
  el.addEventListener = function(type, fn) {
    (el._listeners[type] = el._listeners[type] || []).push(fn);
  };
  el.appendChild = function(child) { el.children.push(child); return child; };
  return el;
}

function Option(text, value) {
  this.text = text;
  this.value = value;
  this.defaultSelected = false;
}

// A <select> with the options (and the "selected" and "disabled" attributes)
// Python rendered; value and selectedIndex behave as a browser's do.
function __select(id, spec) {
  var el = __element(id, 'select');
  var index = spec.options.length ? 0 : -1;
  el.options = spec.options.map(function(o, i) {
    var opt = new Option(o.text, o.value);
    opt.defaultSelected = !!o.selected;
    if (opt.defaultSelected) { index = i; }
    return opt;
  });
  Object.defineProperty(el, 'selectedIndex', {
    get: function() { return index; },
    set: function(i) { index = i; }
  });
  Object.defineProperty(el, 'value', {
    get: function() { return index >= 0 && index < el.options.length ? el.options[index].value : ''; },
    set: function(v) {
      index = -1;
      for (var i = 0; i < el.options.length; i++) {
        if (el.options[i].value === String(v)) { index = i; break; }
      }
    }
  });
  el.add = function(opt) { el.options.push(opt); };
  el.remove = function(i) {
    el.options.splice(i, 1);
    // As a browser does: losing the selected option selects the first one
    if (i === index) { index = el.options.length ? 0 : -1; }
    else if (i < index) { index -= 1; }
  };
  el.disabled = !!spec.disabled;
  return el;
}

var document = {
  getElementById: function(id) {
    if (!__elements[id]) {
      if (__STRICT && !__IDS[id]) { return null; }
      __elements[id] = __element(id);
    }
    return __elements[id];
  },
  createElement: function(tag) { return __element('', tag); }
};

function __target(gd) { return typeof gd === 'string' ? gd : gd.id; }

var Plotly = {
  react: function(gd, data, layout) {
    gd.data = data;
    gd.layout = layout;
    __reacts.push({id: gd.id, data: data, layout: layout});
  },
  // Recorded apart from the redraws: an update or a restyle changes a chart
  // in place, and a test reads them separately
  update: function(gd, dataUpdate, layoutUpdate, traces) {
    __updates.push({id: __target(gd), kind: 'update', data: dataUpdate,
                    layout: layoutUpdate, traces: traces});
  },
  restyle: function(gd, update, traces) {
    __updates.push({id: __target(gd), kind: 'restyle', data: update, traces: traces});
  },
  Plots: {resize: function() {}}
};
var window = {Plotly: Plotly, DecompressionStream: function() {}, Response: function() {},
              Blob: function() {}};

// The page scripts' inflate(el): the block of that element, already inflated
// (a fresh copy each time, as a real inflate parses one). A missing element
// throws here, as atob(el.textContent) would; so does a damaged block. A
// deferred block's inflate settles only at its ["resolve"] / ["reject"] step.
function __inflate(el) {
  __inflated.push(el.id);
  if (__DAMAGED[el.id] || !(el.id in __BLOCKS)) {
    throw new Error('damaged block ' + el.id);
  }
  if (__DEFERRED[el.id]) {
    return new Promise(function(resolve, reject) {
      (__PENDING[el.id] = __PENDING[el.id] || []).push({resolve: resolve, reject: reject});
    });
  }
  return Promise.resolve(JSON.parse(JSON.stringify(__BLOCKS[el.id])));
}

// The page as Python rendered it: its selects and the charts the script
// redraws, and, in strict mode, which ids it holds at all
function __setup(page) {
  __STRICT = !!page.strict;
  (page.ids || []).forEach(function(id) { __IDS[id] = true; });
  Object.keys(page.selects).forEach(function(id) {
    __elements[id] = __select(id, page.selects[id]);
  });
  Object.keys(page.charts).forEach(function(id) {
    var gd = __element(id);
    gd.data = page.charts[id].data;
    gd.layout = page.charts[id].layout;
    __elements[id] = gd;
  });
}

// Everything the script has drawn since the last snapshot, and the state of
// every element it has touched: its text, markup, display, heights and
// colour, a select's value and options, and a table body's rows (each cell's
// text, and each cell's font weight where the script set one)
function __snapshot() {
  var snap = {reacts: __reacts, updates: __updates, inflated: __inflated.slice(),
              text: {}, html: {}, display: {}, heights: {}, ownHeights: {},
              colors: {}, selects: {}, rows: {}, weights: {},
              pending: Object.keys(__PENDING)};
  __reacts = [];
  __updates = [];
  Object.keys(__elements).forEach(function(id) {
    var el = __elements[id];
    if (el._text !== '') { snap.text[id] = el._text; }
    if (el.innerHTML !== '') { snap.html[id] = el.innerHTML; }
    if (el.style.display !== undefined) { snap.display[id] = el.style.display; }
    if (el.parentElement.style.height) { snap.heights[id] = el.parentElement.style.height; }
    if (el.style.height) { snap.ownHeights[id] = el.style.height; }
    if (el.style.color) { snap.colors[id] = el.style.color; }
    if (el.tagName === 'select') {
      snap.selects[id] = {value: el.value, disabled: el.disabled,
                          options: el.options.map(function(o) { return [o.value, o.text]; })};
    }
    if (el.children.length) {
      snap.rows[id] = el.children.map(function(tr) {
        return tr.children.map(function(td) { return td._text; });
      });
      snap.weights[id] = el.children.map(function(tr) {
        return tr.children.map(function(td) { return td.style.fontWeight || ''; });
      });
    }
  });
  return snap;
}

// The filter bar's selects: "wait" lasts until every one of them is enabled
// (which the script does only once the base block and the primary scenario's
// chunk are loaded), or a bounded number of ticks — and, when the explorer's
// script runs too (__WAIT_EXPLORER), until the explorer's selects the page
// holds are enabled as well (once its base block and primary cap's block are
// unpacked)
var __BAR = ['flt-band', 'flt-k', 'flt-cap', 'flt-cat', 'flt-tag'];
var __EXPLORER = ['scn-band-select', 'scn-k-select', 'scn-cap-select', 'scn-metric'];
var __WAIT_EXPLORER = false;

function __barEnabled() {
  var bar = __BAR.every(function(id) {
    var el = document.getElementById(id);
    return !el || !el.disabled;
  });
  return bar && (!__WAIT_EXPLORER || __EXPLORER.every(function(id) {
    var el = __elements[id];
    return !el || !el.disabled;
  }));
}

// Let every pending promise run (a bounded number of microtask ticks), then
// go on.
function __spin(ticks, next) {
  var left = ticks;
  (function spin() {
    if (--left <= 0) {
      next();
    } else {
      Promise.resolve().then(spin);
    }
  })();
}

// Steps: ["wait"] (above), ["settle"] (let every pending promise run — a
// bounded number of microtask ticks), ["set", id, value], ["fire", id] (a
// change event), ["zoom", id] (a reader zooming that chart's x axis),
// ["hide", id] (a reader hiding its first trace from the legend), ["select",
// id] (a reader box-selecting two of its first trace's points), ["repair",
// id] (a damaged block reads again), ["resolve", id] / ["reject", id] (a
// deferred block's waiting inflates finish, or fail as a damaged block's
// would — after a settle, so every load already started has reached its
// inflate, and followed by one; nothing waiting is an error), ["call", name,
// args] (a page script's window function, called as another script would —
// the filter's window.dashScenarioSelect call, with any labels), ["snap",
// name]. Emits every snapshot, as JSON, once the steps are done.
function __step(steps, i) {
  if (i >= steps.length) {
    __emit(JSON.stringify(__snaps));
    return;
  }
  var s = steps[i];
  if (s[0] === 'resolve' || s[0] === 'reject') {
    __spin(200, function() {
      var waiting = __PENDING[s[1]] || [];
      if (!waiting.length) { throw new Error('no inflate of ' + s[1] + ' is waiting'); }
      delete __PENDING[s[1]];
      waiting.forEach(function(w) {
        if (s[0] === 'resolve') { w.resolve(JSON.parse(JSON.stringify(__BLOCKS[s[1]]))); }
        else { w.reject(new Error('damaged block ' + s[1])); }
      });
      __spin(200, function() { __step(steps, i + 1); });
    });
    return;
  }
  if (s[0] === 'wait') {
    var ticks = 0;
    (function spin() {
      if (__barEnabled() || ++ticks > 1000) {
        __step(steps, i + 1);
      } else {
        Promise.resolve().then(spin);
      }
    })();
    return;
  }
  if (s[0] === 'settle') {
    __spin(200, function() { __step(steps, i + 1); });
    return;
  }
  if (s[0] === 'set') {
    document.getElementById(s[1]).value = s[2];
  } else if (s[0] === 'fire') {
    (document.getElementById(s[1])._listeners.change || []).forEach(function(fn) { fn(); });
  } else if (s[0] === 'zoom') {
    var gd = document.getElementById(s[1]);
    gd.layout = JSON.parse(JSON.stringify(gd.layout));
    gd.layout.xaxis = Object.assign({}, gd.layout.xaxis || {}, {range: [0, 1], autorange: false});
  } else if (s[0] === 'hide') {
    document.getElementById(s[1]).data[0].visible = 'legendonly';
  } else if (s[0] === 'select') {
    document.getElementById(s[1]).data[0].selectedpoints = [0, 1];
  } else if (s[0] === 'repair') {
    delete __DAMAGED[s[1]];
  } else if (s[0] === 'call') {
    window[s[1]].apply(null, s[2]);
  } else if (s[0] === 'snap') {
    __snaps[s[1]] = __snapshot();
  }
  __step(steps, i + 1);
}
