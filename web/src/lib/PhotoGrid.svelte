<script>
  import { api } from "./api.js";
  import { createEventDispatcher } from "svelte";
  export let photos = [];
  export let cols = 5;
  const dispatch = createEventDispatcher();
</script>

{#if photos.length === 0}
  <p class="muted">No photos to show.</p>
{:else}
  <div class="grid" style="--cols:{cols}">
    {#each photos as p (p.id)}
      <div class="cell" class:missing={p.exists === false}
           on:click={() => p.exists !== false && dispatch("select", p.id)}
           on:keydown={(e) => e.key === "Enter" && dispatch("select", p.id)}
           role="button" tabindex="0">
        {#if p.exists === false}
          <div class="gone">🚫<br /><small>{p.filename}</small></div>
        {:else}
          <img src={api.thumbUrl(p.id)} alt={p.filename} loading="lazy" />
          {#if p.caption}<div class="cap">{p.caption}</div>{/if}
        {/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(var(--cols), 1fr);
    gap: 8px;
  }
  .cell {
    position: relative; border-radius: 8px; overflow: hidden;
    background: var(--surface); aspect-ratio: 1; cursor: pointer;
    border: 1px solid var(--border);
  }
  .cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .cell:hover img { filter: brightness(1.1); }
  .cap {
    position: absolute; bottom: 0; left: 0; right: 0;
    font-size: 10px; padding: 6px 4px 4px;
    background: linear-gradient(transparent, rgba(0,0,0,.75));
    color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .cell.missing { cursor: default; border-style: dashed; border-color: var(--danger); }
  .gone { display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; color: var(--danger); font-size: 11px; text-align: center; }
</style>
