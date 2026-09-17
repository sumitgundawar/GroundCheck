// Applies the saved theme before the page paints, so it never flashes the
// wrong colours. "system" follows the operating system's light or dark mode.
(function () {
  var KEY = "gc-theme";
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function stored() {
    try { return localStorage.getItem(KEY) || "system"; } catch (e) { return "system"; }
  }
  function apply(choice) {
    var root = document.documentElement;
    if (choice === "light" || choice === "dark") root.setAttribute("data-theme", choice);
    else root.removeAttribute("data-theme");
  }
  function announce() { document.dispatchEvent(new CustomEvent("themechange")); }

  apply(stored());
  window.gcTheme = {
    get: stored,
    set: function (choice) {
      try { localStorage.setItem(KEY, choice); } catch (e) { /* private browsing: apply for this page only */ }
      apply(choice);
      announce();
    },
    isDark: function () {
      var forced = document.documentElement.getAttribute("data-theme");
      return forced ? forced === "dark" : media.matches;
    },
  };
  media.addEventListener("change", function () { if (stored() === "system") announce(); });
})();
