const storageKey = "theme-preference";

const theme = {
  value: getColorPreference(),
};

function onClick() {
  theme.value = theme.value === "light" ? "dark" : "light";

  setPreference();
}

function getColorPreference() {
  if (localStorage.getItem(storageKey)) return localStorage.getItem(storageKey);
  else
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
}

function setPreference() {
  localStorage.setItem(storageKey, theme.value);
  reflectPreference();
}

function reflectPreference() {
  document.firstElementChild.setAttribute("data-theme", theme.value);

  document
    .querySelector("#theme-toggle")
    ?.setAttribute("aria-label", theme.value);
}

reflectPreference();

window.onload = () => {
  reflectPreference();

  document.querySelector("#theme-toggle").addEventListener("click", onClick);
};

window
  .matchMedia("(prefers-color-scheme: dark")
  .addEventListener("change", ({ matches: isDark }) => {
    theme.value = isDark ? "dark" : "light";
    setPreference();
  });
