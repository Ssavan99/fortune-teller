/* Dark-first theme toggle, shared by both pages. Applied synchronously (this script is loaded
   in <head>, not deferred) so the correct theme is set before first paint -- no flash of the
   wrong theme. The button-wiring half waits for DOMContentLoaded since it needs the button
   element to exist. */
(function () {
  "use strict";
  var KEY = "fortune-teller-theme";

  function apply(theme) {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
  }

  var stored = null;
  try {
    stored = localStorage.getItem(KEY);
  } catch (e) {
    /* localStorage can throw in locked-down contexts; theme just falls back to system/dark */
  }
  apply(stored);

  var SUN =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
    'stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4' +
    "M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4\"/></svg>";
  var MOON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4 ' +
    'a7 7 0 0 0 10.5 10.5Z"/></svg>';

  window.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;

    function isDark() {
      var explicit = document.documentElement.getAttribute("data-theme");
      if (explicit) return explicit === "dark";
      return !window.matchMedia("(prefers-color-scheme: light)").matches; // dark-first default
    }
    function refresh() {
      btn.innerHTML = isDark() ? SUN : MOON;
      btn.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
    }
    refresh();
    btn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      apply(next);
      try {
        localStorage.setItem(KEY, next);
      } catch (e) {
        /* best-effort persistence only */
      }
      refresh();
    });
  });
})();
