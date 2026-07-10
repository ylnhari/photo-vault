<script>
  // Inline, section-scoped error line: renders only when the page's single
  // error state belongs to THIS section (scope === at), so failures show up
  // next to the button that caused them instead of in a banner at the top
  // of the page the user may have scrolled away from.
  export let err;
  export let at;
  export let scope;
  export let onclear = null;
</script>

{#if err && at === scope}
  <p class="errline" role="alert">
    ⚠️ {err}
    {#if onclear}<button class="x" on:click={onclear} aria-label="Dismiss">×</button>{/if}
  </p>
{/if}

<style>
  .errline {
    display: flex; align-items: baseline; gap: 8px;
    margin: 10px 0 0; padding: 8px 12px; font-size: 13px;
    color: var(--danger);
    background: color-mix(in srgb, var(--danger) 10%, var(--surface2));
    border: 1px solid color-mix(in srgb, var(--danger) 35%, var(--border));
    border-radius: 10px;
  }
  .x { margin-left: auto; background: none; border: none; cursor: pointer;
       color: var(--danger); font-size: 14px; padding: 0 2px; }
</style>
