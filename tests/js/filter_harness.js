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
// rendering or layout: Plotly.react redraws a chart, and Plotly.update,
// restyle and relayout apply their changes to the chart's data and layout
// the way Plotly does (restyle and update honouring their traces argument)
// and are recorded apart from the redraws (an axis set to autorange keeps
// its last range here, where Plotly would recompute it); Plotly.Plots.resize
// does nothing here. ES5 plus Promise and Object.assign, which both runtimes
// have.
//
// Buttons (page.buttons): each is created with the disabled state Python
// rendered, a ["click", id] step calls its click listeners unless it is
// disabled (as a browser ignores a click on a disabled button), and
// window.open records each call — {url, target, features} — in __opened and
// opens nothing (it returns null, as a "noopener" open does).
//
// Check-box menus (page.menus): the filter bar's Category and Tag menus,
// each a <details> element of check boxes (__menu below). A ["tick", id] step
// is a reader clicking a box: it flips, and its change event runs the box's
// listeners and then the menu's, unless the box is disabled. ["open", id]
// opens a menu, ["outside", id] is a click somewhere on the page (on the
// element named, or on nothing a script knows when id is ""), and ["key",
// name] a key pressed, both handed to the listeners the script put on the
// document. A snapshot's "menus" holds each menu's button text, whether it
// is open, its class, and each row's text, tick and whether it is hidden.
//
// Strict mode (page.strict): getElementById returns null for an id the page
// does not hold, as a browser's does, so a script that dereferences a missing
// element fails the run. Otherwise any id asked for is created on the fly.
//
// Deferred blocks (__DEFERRED): their inflate waits for a ["resolve", id] or
// ["reject", id] step, as a slow DecompressionStream's would, so every order
// in which choices and arriving chunks can interleave can be driven on
// purpose. Every other block inflates at once.
//
// Sidecar files (__SIDECARS): a <script> element the page appends to
// document.head loads the file its src names — a chunk the Sell select reads
// from beside the page — as a browser would: asynchronously, the file's one
// call to window.__dashChunk running with document.currentScript set to that
// element (handing over the chunk, already inflated, which the script's
// inflateText — replaced by __inflateText below — passes through), then the
// element's onload. A src not in __SIDECARS fires onerror instead; one in
// __SIDECAR_EMPTY runs without handing anything over (onload alone); one in
// __SIDECAR_DEFERRED waits for a ["resolve_file", src] or ["reject_file", src]
// step. Every src loaded is recorded, in order (a snapshot's "files").

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
// Every window.open call since the last snapshot, in order
var __opened = [];
// The sidecar files the page may load (src -> chunk), the ones that hand
// nothing over, the ones whose load waits for a step, the loads waiting, by
// src, and every src loaded, in order
var __SIDECARS = {};
var __SIDECAR_EMPTY = {};
var __SIDECAR_DEFERRED = {};
var __FILE_PENDING = {};
var __files = [];

function __escape(text) {
  return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function __element(id, tag) {
  var el = {id: id, tagName: tag || 'div', style: {}, disabled: false,
            parentElement: {style: {}}, data: null, layout: null, children: [],
            _text: '', _html: '', _listeners: {}};
  // A page element's markup is a plain backing field a script may set; a
  // created element redefines it (createElement, below)
  Object.defineProperty(el, 'innerHTML', {
    configurable: true,
    get: function() { return el._html; },
    set: function(v) { el._html = String(v); }
  });
  // As a browser's: setting textContent replaces the children and makes the
  // markup the text, escaped — written to the backing field, so a created
  // element (whose innerHTML setter throws) takes textContent too
  Object.defineProperty(el, 'textContent', {
    get: function() { return el._text; },
    set: function(v) { el._text = String(v); el.children = []; el._html = __escape(v); }
  });
  el.addEventListener = function(type, fn) {
    (el._listeners[type] = el._listeners[type] || []).push(fn);
  };
  el.appendChild = function(child) { el.children.push(child); return child; };
  el._attrs = {};
  el.setAttribute = function(name, value) { el._attrs[name] = String(value); };
  el.getAttribute = function(name) { return name in el._attrs ? el._attrs[name] : null; };
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

// A check-box menu (the filter bar's Category and Tag menus: a <details>
// element) with the boxes, texts and attributes Python rendered: its button
// (<summary>, "<id>-label"), the box that clears it ("<id>-all") and, per
// row i, the box ("<id>-<i>"), its text ("<id>-<i>-text") and the row itself
// ("<id>-<i>-row", which a script hides). A box has "checked" and "disabled";
// a change event on a box runs the box's own listeners and then the menu's
// (it bubbles), each handed {target: the box}. The menu answers
// querySelectorAll with its boxes (the clearing box first) and contains()
// for itself and its parts. For the tests' sake it also reads like a
// <select>: "disabled" is its clearing box's, and "value" is the ticked row
// indexes joined by commas ("" for none) — setting it ticks exactly those
// rows, as a browser restoring a reader's old ticks would.
var __MENUS = {};
function __checkbox(id, spec) {
  var el = __element(id, 'input');
  el.type = 'checkbox';
  el.checked = !!spec.checked;
  el.disabled = !!spec.disabled;
  __elements[id] = el;
  return el;
}
function __menu(id, spec) {
  var el = __element(id, 'details');
  el.open = false;
  el.className = spec.className || '';
  var label = __element(id + '-label', 'summary');
  label.textContent = spec.label;
  __elements[id + '-label'] = label;
  var all = __checkbox(id + '-all', spec.all);
  var parts = [el, label, all];
  var rows = spec.rows.map(function(row, i) {
    var box = __checkbox(id + '-' + i, row);
    var text = __element(id + '-' + i + '-text', 'span');
    text.textContent = row.text;
    __elements[id + '-' + i + '-text'] = text;
    var line = __element(id + '-' + i + '-row', 'label');
    line.hidden = !!row.hidden;
    __elements[id + '-' + i + '-row'] = line;
    parts.push(box, text, line);
    return {box: box, text: text, line: line};
  });
  el.querySelectorAll = function() {
    return [all].concat(rows.map(function(r) { return r.box; }));
  };
  el.contains = function(other) { return parts.indexOf(other) >= 0; };
  Object.defineProperty(el, 'disabled', {
    get: function() { return all.disabled; },
    set: function() {}
  });
  Object.defineProperty(el, 'value', {
    get: function() {
      var out = [];
      rows.forEach(function(r, i) { if (r.box.checked) { out.push(i); } });
      return out.join(',');
    },
    set: function(v) {
      var wanted = String(v) === '' ? [] : String(v).split(',');
      rows.forEach(function(r, i) { r.box.checked = wanted.indexOf(String(i)) >= 0; });
      all.checked = !wanted.length;
    }
  });
  __MENUS[id] = {el: el, label: label, all: all, allText: spec.all.text, rows: rows};
  return el;
}
// A change event on an element: its own listeners, then every menu's that
// holds it (the event bubbles up to the <details>)
function __change(el) {
  var ev = {target: el};
  (el._listeners.change || []).forEach(function(fn) { fn(ev); });
  Object.keys(__MENUS).forEach(function(id) {
    var menu = __MENUS[id].el;
    if (menu !== el && menu.contains(el)) {
      (menu._listeners.change || []).forEach(function(fn) { fn(ev); });
    }
  });
}

// A sidecar file loading: its call to window.__dashChunk runs (unless it
// hands nothing over), with document.currentScript set to the element, then
// the element's onload — or, for a file that cannot be loaded, its onerror
function __runFile(el, fails) {
  var src = el.src;
  if (fails || !(src in __SIDECARS)) {
    if (el.onerror) { el.onerror(new Error('cannot load ' + src)); }
    return;
  }
  if (!__SIDECAR_EMPTY[src]) {
    document.currentScript = el;
    try {
      window.__dashChunk(el, JSON.parse(JSON.stringify(__SIDECARS[src])));
    } finally {
      document.currentScript = null;
    }
  }
  if (el.onload) { el.onload(); }
}

// document.head: a <script> appended to it loads its src, a tick later (or
// at its ["resolve_file"] / ["reject_file"] step, when deferred)
var __HEAD = {children: []};
__HEAD.appendChild = function(el) {
  el.parentNode = __HEAD;
  __HEAD.children.push(el);
  if (el.tagName === 'script') {
    __files.push(el.src);
    if (__SIDECAR_DEFERRED[el.src]) {
      (__FILE_PENDING[el.src] = __FILE_PENDING[el.src] || []).push(el);
    } else {
      Promise.resolve().then(function() { __runFile(el, false); });
    }
  }
  return el;
};
__HEAD.removeChild = function(el) {
  var at = __HEAD.children.indexOf(el);
  if (at >= 0) { __HEAD.children.splice(at, 1); }
  el.parentNode = null;
  return el;
};

// Listeners a script puts on the document itself (a click anywhere, a key),
// by event type; the ["outside"] and ["key"] steps call them
var __DOC_LISTENERS = {};

var document = {
  head: __HEAD,
  currentScript: null,
  addEventListener: function(type, fn) {
    (__DOC_LISTENERS[type] = __DOC_LISTENERS[type] || []).push(fn);
  },
  getElementById: function(id) {
    if (!__elements[id]) {
      if (__STRICT && !__IDS[id]) { return null; }
      __elements[id] = __element(id);
    }
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

function __target(gd) { return typeof gd === 'string' ? gd : gd.id; }
// A chart named by its id, as Plotly accepts one, or the chart itself
function __chart(gd) { return typeof gd === 'string' ? document.getElementById(gd) : gd; }
// One Plotly attribute string ("title.text", "xaxis.autorange") set on an
// object, as Plotly sets it
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
// A restyle's changes, trace by trace, as Plotly applies them: to the traces
// named (one index or an array of them), or every trace when none are; an
// array value holds one entry per trace changed, in that order, cycling
function __restyle(gd, update, traces) {
  var data = gd.data || [];
  var which = (traces === undefined || traces === null)
    ? data.map(function(_, i) { return i; })
    : (Array.isArray(traces) ? traces : [traces]);
  which.forEach(function(ti, j) {
    var trace = data[ti];
    if (!trace) { return; }
    Object.keys(update || {}).forEach(function(key) {
      var v = update[key];
      __assign(trace, key, Array.isArray(v) ? v[j % v.length] : v);
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
  // Recorded apart from the redraws: an update, a restyle or a relayout
  // changes a chart in place — and here, as in Plotly, its data and layout —
  // and a test reads them separately
  update: function(gd, dataUpdate, layoutUpdate, traces) {
    __updates.push({id: __target(gd), kind: 'update', data: dataUpdate,
                    layout: layoutUpdate, traces: traces});
    gd = __chart(gd);
    if (gd) { __restyle(gd, dataUpdate, traces); __relayout(gd, layoutUpdate); }
  },
  restyle: function(gd, update, traces) {
    __updates.push({id: __target(gd), kind: 'restyle', data: update, traces: traces});
    gd = __chart(gd);
    if (gd) { __restyle(gd, update, traces); }
  },
  relayout: function(gd, update) {
    __updates.push({id: __target(gd), kind: 'relayout', layout: update});
    gd = __chart(gd);
    if (gd) { __relayout(gd, update); }
  },
  Plots: {resize: function() {}}
};
var window = {Plotly: Plotly, DecompressionStream: function() {}, Response: function() {},
              Blob: function() {},
              // A new tab the script asks for: recorded, never opened
              open: function(url, target, features) {
                __opened.push({url: url, target: target, features: features});
                return null;
              }};

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

// The filter script's inflateText(text): a sidecar file's chunk, which the
// file handed over already inflated (a fresh copy, as a real inflate parses one)
function __inflateText(text) {
  return Promise.resolve(JSON.parse(JSON.stringify(text)));
}

// The page as Python rendered it: its selects, its buttons, the charts the
// scripts drive, the text of any element a test sets (page.texts) and, in
// strict mode (page.strict), which ids it holds at all
function __setup(page) {
  __STRICT = !!page.strict;
  (page.ids || []).forEach(function(id) { __IDS[id] = true; });
  Object.keys(page.selects).forEach(function(id) {
    // A check-box menu is listed among the selects too (its rows as
    // options, for the tests to read): built as a menu, below
    if (page.selects[id].menu) { return; }
    __elements[id] = __select(id, page.selects[id]);
  });
  Object.keys(page.menus || {}).forEach(function(id) {
    __elements[id] = __menu(id, page.menus[id]);
  });
  Object.keys(page.buttons || {}).forEach(function(id) {
    var button = __element(id, 'button');
    button.disabled = !!page.buttons[id].disabled;
    __elements[id] = button;
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

// Everything the scripts have drawn, changed and opened since the last
// snapshot, and the state of every element they have touched: its text,
// markup, display, heights and colour, a select's value and options, a
// button's disabled state, a table body's rows (each cell's text, and each
// cell's font weight where the script set one), and a chart's layout copied
// as it stands now, since a later step can still change it
function __snapshot() {
  var snap = {reacts: __reacts, updates: __updates, inflated: __inflated.slice(),
              opened: __opened, files: __files.slice(),
              text: {}, html: {}, display: {}, heights: {}, ownHeights: {},
              colors: {}, selects: {}, menus: {}, buttons: {}, rows: {}, weights: {},
              layouts: {}, pending: Object.keys(__PENDING),
              pendingFiles: Object.keys(__FILE_PENDING)};
  __reacts = [];
  __updates = [];
  __opened = [];
  // Each check-box menu: its button's text, whether it is open, greyed
  // (className) and disabled, whether its clearing box is ticked, the row
  // indexes ticked, and every row as [text, ticked, hidden] — and, read as a
  // select, its value (the ticked indexes, comma-joined) and its options
  // (the clearing box, then the rows not hidden)
  Object.keys(__MENUS).forEach(function(id) {
    var m = __MENUS[id], ticked = [], options = [['', m.allText]];
    m.rows.forEach(function(r, i) {
      if (r.box.checked) { ticked.push(i); }
      if (!r.line.hidden) { options.push([String(i), r.text._text]); }
    });
    snap.menus[id] = {label: m.label._text, open: !!m.el.open, className: m.el.className,
                      disabled: m.all.disabled, all: m.all.checked, ticked: ticked,
                      boxesDisabled: m.rows.map(function(r) { return r.box.disabled; }),
                      rows: m.rows.map(function(r) {
                        return [r.text._text, r.box.checked, !!r.line.hidden];
                      })};
    snap.selects[id] = {value: m.el.value, disabled: m.all.disabled, options: options};
  });
  Object.keys(__elements).forEach(function(id) {
    var el = __elements[id];
    if (el.layout) { snap.layouts[id] = JSON.parse(JSON.stringify(el.layout)); }
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
    if (el.tagName === 'button') { snap.buttons[id] = el.disabled; }
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
// unpacked; the explorer's Tier floors select only on a page that renders
// it, which it enables with the others). The bar's Tier floors select
// (flt-tier) is not among them: a run with no tier-floors-off view keeps it
// disabled by design, and waiting on it would stall every such run.
var __BAR = ['flt-band', 'flt-k', 'flt-cap', 'flt-cat', 'flt-tag'];
var __EXPLORER = ['scn-band-select', 'scn-k-select', 'scn-cap-select', 'scn-metric',
                  'scn-tier-select'];
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

// Steps: ["wait"] (above; at once on a page with no bar), ["settle"] (let
// every pending promise run — a bounded number of microtask ticks), ["set",
// id, value], ["fire", id] (a change event), ["zoom", id] (a reader zooming
// that chart's x axis), ["hide", id] (a reader hiding its first trace from
// the legend), ["select", id] (a reader box-selecting two of its first
// trace's points), ["repair", id] (a damaged block reads again), ["resolve",
// id] / ["reject", id] (a deferred block's waiting inflates finish, or fail as
// a damaged block's would — after a settle, so every load already started has
// reached its inflate, and followed by one; nothing waiting is an error),
// ["resolve_file", src] / ["reject_file", src] (a deferred sidecar file's
// waiting loads finish, or fail to load, the same way),
// ["call", name, args] (a page script's window function, called as another
// script would — the filter's window.dashScenarioSelect call, with any
// labels), ["click", id] (a reader clicking that element: its click
// listeners run unless it is disabled), ["tick", id] (a reader clicking a
// check box of a menu), ["open", id] (a reader opening a menu), ["outside",
// id] (a click elsewhere on the page), ["key", name] (a key pressed),
// ["snap", name]. Emits every snapshot, as JSON, once the steps are done.
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
  if (s[0] === 'resolve_file' || s[0] === 'reject_file') {
    __spin(200, function() {
      var loading = __FILE_PENDING[s[1]] || [];
      if (!loading.length) { throw new Error('no load of ' + s[1] + ' is waiting'); }
      delete __FILE_PENDING[s[1]];
      loading.forEach(function(el) { __runFile(el, s[0] === 'reject_file'); });
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
  } else if (s[0] === 'tick') {
    // A reader clicking a check box: it flips, then its change event runs —
    // unless the box is disabled, which a browser leaves alone
    var box = document.getElementById(s[1]);
    if (!box.disabled) {
      box.checked = !box.checked;
      __change(box);
    }
  } else if (s[0] === 'open') {
    document.getElementById(s[1]).open = true;
  } else if (s[0] === 'outside') {
    // A click somewhere on the page: on the element named, or on nothing a
    // script knows
    var clickedOn = s[1] ? document.getElementById(s[1]) : {};
    (__DOC_LISTENERS.click || []).forEach(function(fn) { fn({target: clickedOn}); });
  } else if (s[0] === 'key') {
    (__DOC_LISTENERS.keydown || []).forEach(function(fn) { fn({key: s[1]}); });
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
  } else if (s[0] === 'click') {
    var clicked = document.getElementById(s[1]);
    if (!clicked.disabled) {
      (clicked._listeners.click || []).forEach(function(fn) { fn(); });
    }
  } else if (s[0] === 'snap') {
    __snaps[s[1]] = __snapshot();
  }
  __step(steps, i + 1);
}
