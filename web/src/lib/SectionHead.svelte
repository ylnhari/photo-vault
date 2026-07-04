<script>
  // Small, consistent "icon tile + title" header used across Index & Manage
  // so each pipeline stage reads as its own distinct thing instead of an
  // undifferentiated stack of identical grey cards.
  export let icon = "⚙";
  export let color = "#6366f1";
  export let title = "";
  export let id = undefined;
</script>

<div class="section-head" {id} style="--tile:{color}">
  <span class="tile">{icon}</span>
  <div class="section-title">{title}<slot /></div>
</div>

<style>
  .section-head { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .section-title {
    font-size: 15px; font-weight: 700; letter-spacing: .01em; color: var(--text);
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  }
</style>
