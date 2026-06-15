<script>
  export let label;
  export let state = "off";   // "on" | "off" | "warn" | "unknown"
  export let detail = "";
</script>

<div class="pill state-{state}" title={detail}>
  <span class="dot"></span>
  <span class="lbl">{label}</span>
  {#if detail}<span class="detail">{detail}</span>{/if}
</div>

<style>
  .pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 12px; border-radius: 99px;
    background: var(--surface); border: 1px solid var(--border);
    font-size: 13px; line-height: 1;
  }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
  .state-on    .dot { background: var(--success); box-shadow: 0 0 0 3px color-mix(in srgb, var(--success) 25%, transparent); }
  .state-off   .dot { background: var(--danger); }
  .state-warn  .dot { background: var(--warn); box-shadow: 0 0 0 3px color-mix(in srgb, var(--warn) 25%, transparent); }
  .state-on    { border-color: color-mix(in srgb, var(--success) 40%, var(--border)); }
  .state-warn  { border-color: color-mix(in srgb, var(--warn) 40%, var(--border)); }
  .lbl { font-weight: 600; }
  .detail { color: var(--muted); }
</style>
