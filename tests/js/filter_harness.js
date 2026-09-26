// A stand-in for the browser, just large enough to run the dashboard's
// page-wide filter script (dashboard._FILTER_JS) under a plain JavaScript
// runtime: node (CI) or JavaScriptCore's jsc (macOS). Driven by
// tests/test_dashboard.py's _run_script, which appends the page's elements,
// the filter data, the script itself and a list of steps. It checks the
// script's own logic — what it draws, writes and enables for each choice —
// not a browser's rendering or layout (Plotly.Plots.resize does nothing
// here). ES5 plus Promise and Object.assign, which both runtimes have.

var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __elements = {};
var __reacts = [];
var __snaps = {};

function __element(id, tag) {
  var el = {id: id, tagName: tag || 'div', style: {}, innerHTML: '', disabled: false,
            parentElement: {style: {}}, data: null, layout: null, children: [],
            _text: '', _listeners: {}};
  Object.defineProperty(el, 'textContent', {
    get: function() { return el._text; },
    set: function(v) { el._text = String(v); el.children = []; }
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
    if (!__elements[id]) { __elements[id] = __element(id); }
    return __elements[id];
  },
  createElement: function(tag) { return __element('', tag); }
};

var Plotly = {
  react: function(gd, data, layout) {
    gd.data = data;
    gd.layout = layout;
    __reacts.push({id: gd.id, data: data, layout: layout});
  },
  Plots: {resize: function() {}}
};
var window = {Plotly: Plotly, DecompressionStream: function() {}, Response: function() {},
              Blob: function() {}};

// The page as Python rendered it: its selects and the charts the script redraws
function __setup(page) {
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
// every element it has touched
function __snapshot() {
  var snap = {reacts: __reacts, text: {}, html: {}, display: {}, heights: {}, ownHeights: {},
              selects: {}, rows: {}};
  __reacts = [];
  Object.keys(__elements).forEach(function(id) {
    var el = __elements[id];
    if (el._text !== '') { snap.text[id] = el._text; }
    if (el.innerHTML !== '') { snap.html[id] = el.innerHTML; }
    if (el.style.display !== undefined) { snap.display[id] = el.style.display; }
    if (el.parentElement.style.height) { snap.heights[id] = el.parentElement.style.height; }
    if (el.style.height) { snap.ownHeights[id] = el.style.height; }
    if (el.tagName === 'select') {
      snap.selects[id] = {value: el.value, disabled: el.disabled,
                          options: el.options.map(function(o) { return [o.value, o.text]; })};
    }
    if (el.children.length) {
      snap.rows[id] = el.children.map(function(tr) {
        return tr.children.map(function(td) { return td._text; });
      });
    }
  });
  return snap;
}

// Steps: ["wait"] (until the bar is enabled, or a bounded number of ticks),
// ["set", id, value], ["fire", id] (a change event), ["zoom", id] (a reader
// zooming that chart's x axis), ["hide", id] (a reader hiding its first trace
// from the legend), ["select", id] (a reader box-selecting two of its first
// trace's points), ["snap", name]. Emits every snapshot, as JSON, once the
// steps are done.
function __step(steps, i) {
  if (i >= steps.length) {
    __emit(JSON.stringify(__snaps));
    return;
  }
  var s = steps[i];
  if (s[0] === 'wait') {
    var ticks = 0;
    (function spin() {
      if (!document.getElementById('flt-band').disabled || ++ticks > 1000) {
        __step(steps, i + 1);
      } else {
        Promise.resolve().then(spin);
      }
    })();
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
  } else if (s[0] === 'snap') {
    __snaps[s[1]] = __snapshot();
  }
  __step(steps, i + 1);
}
