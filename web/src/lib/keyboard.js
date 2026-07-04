// Shared Enter/Space "activate" keydown handler for custom clickable elements
// (role="button" divs/imgs that aren't real <button>s), matching native
// button keyboard behavior. Space is prevented so it doesn't also scroll the
// page. Used across AlbumsTab, PeopleTab, Lightbox, IndexTab for consistency.
export function onActivateKey(e, fn) {
  if (e.key === "Enter") {
    fn(e);
  } else if (e.key === " " || e.key === "Spacebar") {
    e.preventDefault();
    fn(e);
  }
}
