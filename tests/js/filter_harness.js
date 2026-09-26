// A stand-in for the browser, just large enough to run the dashboard's
// page-wide filter script (dashboard._FILTER_JS) — and, for the scenario
// explorer's tests, the explorer's own script (dashboard._SCENARIO_EXPLORER_JS)
// before it, in the page's order — under a plain JavaScript runtime: node (CI)
// or JavaScriptCore's jsc (macOS). Driven by tests/test_dashboard.py's
// _run_script, which appends the page's elements, the filter data, the
// script(s) and a list of steps. It checks the scripts' own logic — what they
// draw, write and enable for each choice — not a browser's rendering or
// layout: Plotly.react, restyle, update and relayout apply their changes to
// the chart's data and layout the way Plotly does and are recorded (an axis
// set to autorange keeps its last range here, where Plotly would recompute
// it), and Plotly.Plots.resize only records which chart it was asked to
// resize. By default getElementById hands back an element for any id,
// creating it on first use; a page that lists its ids (page.ids) switches on
// a strict mode in which an id the page does not carry reads null, as in a
// browser. ES5 plus Promise and Object.assign, which both runtimes have.

var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __elements = {};
var __reacts = [];
var __calls = [];
var __snaps = {};
// Strict mode's ids (page.ids), or null: every id is created on first use
var __strict = null;

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
    // Strict mode: an id the page does not carry is not there
    if (__strict && __strict.indexOf(id) < 0) { return null; }
    if (!__elements[id]) { __elements[id] = __element(id); }
    return __elements[id];
  },
  createElement: function(tag) {
    var el = __element('', tag);
    // A browser serialises an element's text escaped (&, <, > and the
    // no-break space), which is what the explorer's esc() reads back; the
    // filter script never reads it. Neither script WRITES a created
    // element's innerHTML — both write its textContent — and this stand-in
    // could not parse one, so a write is an error rather than a silent no-op
    Object.defineProperty(el, 'innerHTML', {
      set: function() {
        throw new Error('filter_harness: a script set a created element\'s innerHTML, '
          + 'which this stand-in cannot parse (write its textContent)');
      },
      get: function() {
        return el._text.replace(/&/g, '&amp;').replace(/</g, '&lt;')
          .replace(/>/g, '&gt;').replace(/ /g, '&nbsp;');
      }
    });
    return el;
  }
};

// A chart named by its id, as Plotly accepts one, or the chart itself
function __chart(gd) { return typeof gd === 'string' ? document.getElementById(gd) : gd; }
// One Plotly attribute string ("title.text", "updatemenus[0].active") set on
// an object, as Plotly sets it
function __assign(obj, path, value) {
  var parts = path.replace(/\[(\d+)\]/g, '.$1').split('.');
  for (var i = 0; i < parts.length - 1; i++) {
    if (obj[parts[i]] === undefined || obj[parts[i]] === null) {
      obj[parts[i]] = /^\d+$/.test(parts[i + 1]) ? [] : {};
    }
    obj = obj[parts[i]];
  }
  obj[parts[parts.length - 1]] = value;
}
// A restyle's changes, trace by trace: an array value holds one entry per trace
function __restyle(gd, update) {
  (gd.data || []).forEach(function(trace, i) {
    Object.keys(update || {}).forEach(function(key) {
      var v = update[key];
      __assign(trace, key, Array.isArray(v) ? v[i % v.length] : v);
    });
  });
}
function __relayout(gd, update) {
  gd.layout = gd.layout || {};
  Object.keys(update || {}).forEach(function(key) { __assign(gd.layout, key, update[key]); });
}

var Plotly = {
  react: function(gd, data, layout) {
    gd.data = data;
    gd.layout = layout;
    __reacts.push({id: gd.id, data: data, layout: layout});
  },
  restyle: function(gd, update) {
    gd = __chart(gd);
    __calls.push({fn: 'restyle', id: gd.id, data: update});
    __restyle(gd, update);
  },
  update: function(gd, dataUpdate, layoutUpdate) {
    gd = __chart(gd);
    __calls.push({fn: 'update', id: gd.id, data: dataUpdate, layout: layoutUpdate});
    __restyle(gd, dataUpdate);
    __relayout(gd, layoutUpdate);
  },
  relayout: function(gd, update) {
    gd = __chart(gd);
    __calls.push({fn: 'relayout', id: gd.id, layout: update});
    __relayout(gd, update);
  },
  Plots: {resize: function(gd) { __calls.push({fn: 'resize', id: __chart(gd).id}); }}
};
var window = {Plotly: Plotly, DecompressionStream: function() {}, Response: function() {},
              Blob: function() {}};

// The page as Python rendered it: its selects, the charts the scripts drive,
// the text of any element a script reads (the explorer's scn-data block) and,
// for strict mode only, every id it carries
function __setup(page) {
  __strict = page.ids || null;
  Object.keys(page.selects).forEach(function(id) {
    __elements[id] = __select(id, page.selects[id]);
  });
  Object.keys(page.charts).forEach(function(id) {
    var gd = __element(id);
    gd.data = page.charts[id].data;
    gd.layout = page.charts[id].layout;
    __elements[id] = gd;
  });
  Object.keys(page.texts || {}).forEach(function(id) {
    document.getElementById(id).textContent = page.texts[id];
  });
}

// Everything the scripts have drawn and called since the last snapshot, and
// the state of every element they have touched — a chart's layout copied as
// it stands now, since a later step can still change it
function __snapshot() {
  var snap = {reacts: __reacts, calls: __calls, text: {}, html: {}, display: {}, heights: {},
              ownHeights: {}, selects: {}, rows: {}, layouts: {}};
  __reacts = [];
  __calls = [];
  Object.keys(__elements).forEach(function(id) {
    var el = __elements[id];
    if (el.layout) { snap.layouts[id] = JSON.parse(JSON.stringify(el.layout)); }
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

// Steps: ["wait"] (until the bar is enabled, or a bounded number of ticks; at
// once on a page with no bar), ["set", id, value], ["fire", id] (a change
// event), ["zoom", id] (a reader
// zooming that chart's x axis), ["hide", id] (a reader hiding its first trace
// from the legend), ["select", id] (a reader box-selecting two of its first
// trace's points), ["menu", id, i] (a reader picking button i of that chart's
// updatemenus, which plotly.js 2.x records as layout.updatemenus[0].active),
// ["snap", name]. Emits every snapshot, as JSON, once the steps are done.
function __step(steps, i) {
  if (i >= steps.length) {
    __emit(JSON.stringify(__snaps));
    return;
  }
  var s = steps[i];
  if (s[0] === 'wait') {
    var ticks = 0;
    (function spin() {
      var bar = document.getElementById('flt-band');
      if (!bar || !bar.disabled || ++ticks > 1000) {
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
  } else if (s[0] === 'menu') {
    document.getElementById(s[1]).layout.updatemenus[0].active = s[2];
  } else if (s[0] === 'snap') {
    __snaps[s[1]] = __snapshot();
  }
  __step(steps, i + 1);
}
