// A stand-in for the browser, just large enough to run the live dashboard's
// tab page script (live_dashboard._SCRIPT) under a plain JavaScript runtime:
// node (CI) or JavaScriptCore's jsc (macOS). Driven by
// tests/test_live_dashboard.py's _run_live_script, which appends the ids the
// page holds (__IDS) and those its markup hides (__HIDDEN), the answers its
// requests get (__ANSWERS), whether the chart library loaded (__NO_PLOTLY),
// the script itself and a list of steps.
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
// setTimeout records each timer ({id, fn, ms}) and runs none of them on its
// own; clearTimeout removes one; a ["timers"] step runs every pending timer,
// oldest first, and clears them. A snapshot lists the delays of the timers
// still pending, so a test can check both the delay and that a timer was
// cancelled. An element can be copied (cloneNode, its id, title,
// attributes and address) and swapped for its copy (its parentNode's
// replaceChild); a snapshot counts the swaps ("replaced").
//
// Frames and messages: every element has a contentWindow, as a frame has,
// whose postMessage records each message sent to the page it holds
// ({frame, message, origin}; a snapshot's "posted" lists those since the
// snapshot before). A copy made by cloneNode gets a window of its own, as a
// new frame does; setting an address keeps the window, as navigating a frame
// does. window.addEventListener('message', fn) keeps fn, and a ["message",
// {origin, source, data}] step calls each kept listener with an event of that
// origin, a copy of that data and, as its source, the contentWindow of the
// element whose id source names (null: a window no element holds; "old:" and
// an id: the window of the element last swapped out under that id, a page
// that is no longer in the tab page). An element whose id is in __HIDDEN
// starts hidden, as the page's markup has it. A snapshot's "loads" counts
// the addresses set on each id's frame.
// ES5 plus Promise, which both runtimes have.

var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __IDS = {};
var __HIDDEN = {};
var __ANSWERS = [];
var __NO_PLOTLY = false;
var __elements = {};
var __fetches = [];
var __plots = [];
var __snaps = {};
var __srcSets = 0;
var __loads = {};
var __posted = [];
// The page's own window: only its message listeners are needed
var __windowListeners = {};
var window = {
  addEventListener: function(type, fn) {
    (__windowListeners[type] = __windowListeners[type] || []).push(fn);
  }
};
// Every element's parent: replacing a child puts the new element where
// getElementById finds it, keeps the old one's window (a message step's
// "old:" source) and counts the swap (a snapshot's "replaced")
var __replaced = 0;
var __retired = {};
var __parent = {
  replaceChild: function(fresh, old) {
    __elements[old.id] = fresh;
    __retired[old.id] = old.contentWindow;
    __replaced += 1;
    return old;
  }
};
var __timers = [];
var __timerId = 0;

function __element(id, tag) {
  var el = {id: id, tagName: tag || 'div', style: {}, hidden: false, disabled: false,
            className: '', title: '', children: [], _text: '', _attrs: {},
            _listeners: {}};
  Object.defineProperty(el, 'textContent', {
    get: function() { return el._text; },
    // As a browser's: setting the text replaces the children
    set: function(v) { el._text = String(v); el.children = []; }
  });
  // A frame's address: setting it is a navigation (counted in __srcSets, and
  // per id in __loads); copying it into a clone, or removing it, is not
  el._src = '';
  Object.defineProperty(el, 'src', {
    get: function() { return el._src; },
    set: function(v) {
      el._src = String(v);
      __srcSets += 1;
      __loads[el.id] = (__loads[el.id] || 0) + 1;
    }
  });
  // The window of the page a frame holds: kept across navigations, new for a copy
  el.contentWindow = {
    postMessage: function(message, origin) {
      __posted.push({frame: el.id, message: JSON.parse(JSON.stringify(message)),
                     origin: origin});
    }
  };
  el.setAttribute = function(name, value) { el._attrs[name] = String(value); };
  el.getAttribute = function(name) { return name in el._attrs ? el._attrs[name] : null; };
  el.removeAttribute = function(name) {
    if (name === 'src') { el._src = ''; } else { delete el._attrs[name]; }
  };
  // A copy of the element alone (cloneNode(false)): its id, tag, title,
  // attributes and address, nothing it holds
  el.cloneNode = function() {
    var copy = __element(el.id, el.tagName);
    copy.title = el.title;
    copy._attrs = JSON.parse(JSON.stringify(el._attrs));
    copy._src = el._src;
    return copy;
  };
  el.parentNode = __parent;
  el.appendChild = function(child) { el.children.push(child); return child; };
  el.addEventListener = function(type, fn) {
    (el._listeners[type] = el._listeners[type] || []).push(fn);
  };
  return el;
}

var document = {
  getElementById: function(id) {
    if (!__IDS[id]) { return null; }
    if (!__elements[id]) {
      __elements[id] = __element(id);
      __elements[id].hidden = !!__HIDDEN[id];
    }
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

// A timer is only recorded; a ["timers"] step is what runs it
function setTimeout(fn, ms) {
  __timerId += 1;
  __timers.push({id: __timerId, fn: fn, ms: ms});
  return __timerId;
}

function clearTimeout(id) {
  __timers = __timers.filter(function(timer) { return timer.id !== id; });
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
// or list's rows (each cell's text, tooltip and left border); every request,
// chart call and message posted to a frame since the last snapshot; the
// addresses set on each frame so far; and the delays of the timers still
// pending
function __snapshot() {
  var snap = {fetches: __fetches, plots: __plots, posted: __posted, srcSets: __srcSets,
              loads: JSON.parse(JSON.stringify(__loads)), el: {},
              timers: __timers.map(function(timer) { return timer.ms; }),
              replaced: __replaced};
  __fetches = [];
  __plots = [];
  __posted = [];
  Object.keys(__elements).forEach(function(id) {
    var el = __elements[id];
    snap.el[id] = {
      text: el._text, className: el.className, hidden: el.hidden, title: el.title,
      // A copy: the attributes as they stand now, not as a later step leaves them
      attrs: JSON.parse(JSON.stringify(el._attrs)), src: el.src, disabled: el.disabled,
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
// ["timers"] (the time passing: every pending timer runs, oldest first, and
// is cleared), ["message", {origin, source, data}] (a page sending the tab
// page a message: see above), ["snap", name]. Emits every snapshot, as
// JSON, once the steps are done.
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
  } else if (s[0] === 'timers') {
    var due = __timers;
    __timers = [];
    due.forEach(function(timer) { timer.fn(); });
  } else if (s[0] === 'message') {
    var sent = s[1];
    var from = sent.source;
    var event = {
      origin: sent.origin,
      source: from === null ? {} :
        from.indexOf('old:') === 0 ? __retired[from.slice(4)] || {} :
        document.getElementById(from).contentWindow,
      data: sent.data === undefined ? undefined : JSON.parse(JSON.stringify(sent.data))
    };
    (__windowListeners.message || []).forEach(function(fn) { fn(event); });
  } else if (s[0] === 'snap') {
    __snaps[s[1]] = __snapshot();
  }
  __step(steps, i + 1);
}
