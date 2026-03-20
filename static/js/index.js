document.addEventListener("DOMContentLoaded", function () {
  var burger = document.querySelector(".navbar-burger");
  var menu = document.getElementById("site-menu");

  if (burger && menu) {
    burger.addEventListener("click", function () {
      burger.classList.toggle("is-active");
      menu.classList.toggle("is-active");
    });

    menu.querySelectorAll(".navbar-item").forEach(function (item) {
      item.addEventListener("click", function () {
        burger.classList.remove("is-active");
        menu.classList.remove("is-active");
      });
    });
  }

  var tabs = Array.from(document.querySelectorAll(".tab-button"));
  var panels = Array.from(document.querySelectorAll(".example-panel"));

  function activateTab(tabName) {
    tabs.forEach(function (tab) {
      var isActive = tab.dataset.tab === tabName;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
    });

    panels.forEach(function (panel) {
      panel.classList.toggle("is-active", panel.id === tabName);
    });
  }

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      activateTab(tab.dataset.tab);
    });
  });
});
