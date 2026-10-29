// Site behaviour: article table of contents, blog topic filter, contact form.
(function () {
  "use strict";

  const escapeHtml = (text) =>
    String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const slugify = (text) =>
    text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");

  /* ---------------------------------------------------------------------------
   * Article table of contents, built from the headings in the post body
   * ------------------------------------------------------------------------ */
  function initToc() {
    const body = document.querySelector(".article-body");
    const nav = document.getElementById("article-toc");
    if (!body || !nav) return;

    const headings = Array.from(body.querySelectorAll("h1, h2, h3")).filter((h) => !h.closest(".footnotes"));
    if (headings.length < 2) return;

    // the top level is whatever the post uses for sections (some posts use #, some ##)
    const topLevel = Math.min(...headings.map((h) => Number(h.tagName[1])));
    const list = document.createElement("ol");
    const links = [];

    headings.forEach((h) => {
      if (!h.id) h.id = slugify(h.textContent);
      const level = Number(h.tagName[1]);
      if (level > topLevel + 1) return;
      const li = document.createElement("li");
      if (level > topLevel) li.className = "is-sub";
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
      links.push({ heading: h, link: a });
    });

    nav.appendChild(list);
    nav.closest(".article-toc__inner").hidden = false;

    if (!("IntersectionObserver" in window)) return;
    let current = null;
    const setActive = (link) => {
      if (current === link) return;
      if (current) current.classList.remove("is-active");
      current = link;
      if (current) current.classList.add("is-active");
    };
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const match = links.find((l) => l.heading === entry.target);
            if (match) setActive(match.link);
          }
        });
      },
      { rootMargin: "-80px 0px -70% 0px" }
    );
    links.forEach((l) => observer.observe(l.heading));
  }

  /* ---------------------------------------------------------------------------
   * Blog index: keyword search plus type, topic, year and sort filters.
   * State lives in the URL (?q=&tag=&type=&year=&sort=) so filtered views can be shared.
   * ------------------------------------------------------------------------ */
  function initBlogFilter() {
    const bar = document.querySelector("[data-blog-filter]");
    if (!bar) return;

    const qInput = bar.querySelector("[data-filter-q]");
    const typeButtons = Array.from(bar.querySelectorAll("[data-filter-type]"));
    const tagButtons = Array.from(bar.querySelectorAll("[data-filter-tag]"));
    const moreSelect = bar.querySelector("[data-filter-more]");
    const yearSelect = bar.querySelector("[data-filter-year]");
    const sortSelect = bar.querySelector("[data-filter-sort]");
    const resets = Array.from(document.querySelectorAll("[data-filter-reset]"));
    const barReset = bar.querySelector("[data-filter-reset]");
    const count = bar.querySelector("[data-filter-count]");
    const items = Array.from(document.querySelectorAll("[data-filter-item]"));
    const sections = Array.from(document.querySelectorAll("[data-filter-section]"));
    const archive = document.querySelector("[data-archive]");
    const total = items.filter((i) => i.dataset.kind === "archive").length;
    const knownTags = new Set(
      tagButtons.map((b) => b.dataset.filterTag).concat(moreSelect ? Array.from(moreSelect.options, (o) => o.value) : [])
    );

    const state = { q: "", tag: "", type: "", year: "", sort: "new" };

    // every word must appear somewhere in the item's text
    const matchesQuery = (el) => {
      if (!state.q) return true;
      const text = el.dataset.search || "";
      return state.q.split(/\s+/).every((word) => text.includes(word));
    };
    const matchesTag = (el) => !state.tag || (el.dataset.tags || "").split("|").includes(state.tag);
    const matchesYear = (el) => !state.year || (el.dataset.year || "").split("|").includes(state.year);
    const matchesType = (el) => !state.type || el.dataset.type === state.type;

    const sortChildren = (parent, selector, key) => {
      const children = Array.from(parent.querySelectorAll(selector)).filter((c) => c.parentElement === parent);
      children.sort((a, b) => (state.sort === "old" ? key(a) - key(b) : key(b) - key(a)));
      children.forEach((c) => parent.appendChild(c));
    };

    // after a filter change, jump back to the top of the results if we're scrolled past them
    const revealResults = () => {
      const first = sections.find((s) => !s.hasAttribute("data-filter-hidden"));
      if (!first) return;
      const offset = bar.getBoundingClientRect().bottom + 16;
      const top = first.getBoundingClientRect().top + window.scrollY - offset;
      if (window.scrollY > top) window.scrollTo({ top, behavior: "smooth" });
    };

    const apply = (updateUrl) => {
      const active = Boolean(state.q || state.tag || state.type || state.year);

      items.forEach((item) => {
        let match = matchesType(item) && matchesTag(item) && matchesYear(item) && matchesQuery(item);

        // a series card stays visible when any of its parts matches, and dims the parts that don't
        if (item.dataset.kind === "series") {
          const parts = Array.from(item.querySelectorAll("[data-part]"));
          const partMatches = parts.map((part) => matchesTag(part) && matchesYear(part) && matchesQuery(part));
          const anyPart = partMatches.some(Boolean);
          const cardMatchesText = matchesQuery(item) && !state.tag && !state.year;
          match = matchesType(item) && (anyPart || cardMatchesText);
          // point at the matching parts; if only the series blurb matched, show every part normally
          parts.forEach((part, i) => part.classList.toggle("is-dim", active && anyPart && !partMatches[i]));
        }

        item.toggleAttribute("data-filter-hidden", !match);
      });

      sections.forEach((section) => {
        const own = Array.from(section.querySelectorAll("[data-filter-item]"));
        const anyVisible = own.some((el) => !el.hasAttribute("data-filter-hidden"));
        const empty = section.querySelector("[data-filter-empty]");

        if (section.dataset.sectionType) {
          // while filtering, the Series / Articles sections only show up when they have something to show
          const wrongType = state.type && section.dataset.sectionType !== state.type;
          section.toggleAttribute("data-filter-hidden", Boolean(wrongType || (active && !anyVisible)));
        } else if (empty) {
          // the archive always lists every match, and says so when there are none
          empty.hidden = anyVisible;
        }
      });

      // hide year groups in the archive that end up empty
      document.querySelectorAll("[data-archive-year]").forEach((group) => {
        const visible = group.querySelector("[data-filter-item]:not([data-filter-hidden])");
        group.toggleAttribute("data-filter-hidden", !visible);
      });

      // ordering
      document.querySelectorAll(".post-rows").forEach((list) => sortChildren(list, "li", (el) => Number(el.dataset.date || 0)));
      if (archive) {
        sortChildren(archive, "[data-archive-year]", (el) => Number(el.dataset.archiveYear));
        archive.querySelectorAll("[data-archive-year] ul").forEach((list) => sortChildren(list, "li", (el) => Number(el.dataset.date || 0)));
      }

      // controls reflect state
      typeButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filterType === state.type)));
      tagButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filterTag === state.tag)));
      if (moreSelect) {
        const inMore = Array.from(moreSelect.options).some((o) => o.value && o.value === state.tag);
        moreSelect.value = inMore ? state.tag : "";
        moreSelect.classList.toggle("is-active", inMore);
      }
      if (yearSelect) yearSelect.value = state.year;
      if (sortSelect) sortSelect.value = state.sort;
      if (qInput && qInput.value.trim().toLowerCase() !== state.q) qInput.value = state.q;
      if (barReset) barReset.hidden = !active;

      if (count) {
        const shown = items.filter((i) => i.dataset.kind === "archive" && !i.hasAttribute("data-filter-hidden")).length;
        count.textContent = active ? `${shown} of ${total} post${total === 1 ? "" : "s"}` : `${total} post${total === 1 ? "" : "s"}`;
      }

      if (updateUrl) {
        const url = new URL(window.location.href);
        ["q", "tag", "type", "year"].forEach((k) => (state[k] ? url.searchParams.set(k, state[k]) : url.searchParams.delete(k)));
        state.sort === "old" ? url.searchParams.set("sort", "old") : url.searchParams.delete("sort");
        window.history.replaceState({}, "", url);
      }
    };

    // wire up controls
    let typingTimer;
    if (qInput) {
      qInput.addEventListener("input", () => {
        clearTimeout(typingTimer);
        typingTimer = setTimeout(() => {
          state.q = qInput.value.trim().toLowerCase();
          apply(true);
        }, 120);
      });
      qInput.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          qInput.value = "";
          state.q = "";
          apply(true);
        }
      });
      // "/" jumps to the search box, like most docs sites
      document.addEventListener("keydown", (e) => {
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) || document.activeElement.isContentEditable;
        if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) {
          e.preventDefault();
          qInput.focus();
        }
      });
    }
    typeButtons.forEach((b) =>
      b.addEventListener("click", () => {
        state.type = b.dataset.filterType;
        apply(true);
        revealResults();
      })
    );
    tagButtons.forEach((b) =>
      b.addEventListener("click", () => {
        state.tag = state.tag === b.dataset.filterTag ? "" : b.dataset.filterTag;
        apply(true);
        revealResults();
      })
    );
    if (moreSelect)
      moreSelect.addEventListener("change", () => {
        state.tag = moreSelect.value;
        apply(true);
        revealResults();
      });
    if (yearSelect)
      yearSelect.addEventListener("change", () => {
        state.year = yearSelect.value;
        apply(true);
        revealResults();
      });
    if (sortSelect)
      sortSelect.addEventListener("change", () => {
        state.sort = sortSelect.value;
        apply(true);
      });
    resets.forEach((r) =>
      r.addEventListener("click", () => {
        Object.assign(state, { q: "", tag: "", type: "", year: "" });
        apply(true);
        revealResults();
      })
    );

    // initial state from the URL (post pages link here with ?tag=)
    const params = new URLSearchParams(window.location.search);
    state.q = (params.get("q") || "").trim().toLowerCase();
    const tag = (params.get("tag") || "").toLowerCase();
    state.tag = knownTags.has(tag) ? tag : "";
    state.type = ["series", "article"].includes(params.get("type")) ? params.get("type") : "";
    state.year = params.get("year") || "";
    state.sort = params.get("sort") === "old" ? "old" : "new";
    apply(false);
  }

  /* ---------------------------------------------------------------------------
   * Contact form: validate, then post to Web3Forms or Formspree without leaving the page
   * ------------------------------------------------------------------------ */
  function initContactForm() {
    const form = document.querySelector("[data-contact-form]");
    if (!form) return;

    const status = document.querySelector("[data-contact-status]");
    const submit = form.querySelector("button[type=submit]");
    const submitLabel = submit ? submit.innerHTML : "";
    const provider = form.dataset.provider;
    const endpoint = form.action;

    const showStatus = (kind, html) => {
      if (!status) return;
      status.className = `contact-status is-${kind}`;
      status.innerHTML = html;
      status.hidden = false;
      status.focus();
    };

    const setError = (field, message) => {
      const holder = field.closest(".contact-field");
      let el = holder && holder.querySelector(".contact-error");
      if (message) {
        field.setAttribute("aria-invalid", "true");
        if (holder && !el) {
          el = document.createElement("span");
          el.className = "contact-error";
          el.id = `${field.id}-error`;
          holder.appendChild(el);
          field.setAttribute("aria-describedby", el.id);
        }
        if (el) el.textContent = message;
      } else {
        field.removeAttribute("aria-invalid");
        if (el) el.remove();
      }
    };

    const validate = () => {
      let firstInvalid = null;
      form.querySelectorAll("input[required], textarea[required]").forEach((field) => {
        let message = "";
        const value = field.value.trim();
        if (!value) message = "This one's needed.";
        else if (field.type === "email" && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) message = "That email doesn't look right.";
        else if (field.minLength > 0 && value.length < field.minLength) message = `A little more detail, please (${field.minLength}+ characters).`;
        setError(field, message);
        if (message && !firstInvalid) firstInvalid = field;
      });
      if (firstInvalid) firstInvalid.focus();
      return !firstInvalid;
    };

    form.querySelectorAll("input, textarea").forEach((field) => {
      field.addEventListener("input", () => {
        if (field.getAttribute("aria-invalid") === "true") setError(field, "");
      });
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (status) status.hidden = true;
      if (!validate()) return;

      // honeypot: real people never fill this in
      const trap = form.querySelector("[name=botcheck], [name=_gotcha]");
      // (a checkbox's value is always "on", so only its checked state counts)
      if (trap && (trap.type === "checkbox" ? trap.checked : trap.value)) return;

      const data = Object.fromEntries(new FormData(form).entries());
      if (provider === "web3forms") {
        data.from_name = data.name;
        if (data.topic) data.subject = `${form.dataset.subject} · ${data.topic}`;
      }

      if (submit) {
        submit.disabled = true;
        submit.innerHTML = "Sending…";
      }

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(data),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok || result.success === false) {
          throw new Error(result.message || (result.errors && result.errors.map((e) => e.message).join(", ")) || "The form service refused the message.");
        }
        form.reset();
        showStatus("success", "<strong>Message sent.</strong> Thanks for reaching out, I'll get back to you soon.");
      } catch (error) {
        const fallback = form.dataset.fallback;
        showStatus(
          "error",
          `<strong>That didn't go through.</strong> ${escapeHtml(error.message || "")}${fallback ? ` You can also reach me on <a href="${fallback}">LinkedIn</a>.` : ""}`
        );
      } finally {
        if (submit) {
          submit.disabled = false;
          submit.innerHTML = submitLabel;
        }
      }
    });
  }

  const ready = (fn) => (document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", fn) : fn());
  ready(() => {
    initToc();
    initBlogFilter();
    initContactForm();
  });
})();
