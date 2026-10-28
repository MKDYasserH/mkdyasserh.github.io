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
   * Blog index: filter series, articles and archive rows by topic
   * ------------------------------------------------------------------------ */
  function initBlogFilter() {
    const bar = document.querySelector("[data-blog-filter]");
    if (!bar) return;

    const buttons = Array.from(bar.querySelectorAll("[data-filter-tag]"));
    const items = Array.from(document.querySelectorAll("[data-filter-item]"));
    const sections = Array.from(document.querySelectorAll("[data-filter-section]"));
    const count = bar.querySelector("[data-filter-count]");

    const apply = (tag, updateUrl) => {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filterTag === tag)));

      let shown = 0;
      items.forEach((item) => {
        const tags = (item.dataset.tags || "").split("|");
        const match = !tag || tags.includes(tag);
        item.toggleAttribute("data-filter-hidden", !match);
        if (match && item.dataset.kind === "archive") shown += 1;
      });

      sections.forEach((section) => {
        const own = section.querySelectorAll("[data-filter-item]");
        const empty = section.querySelector("[data-filter-empty]");
        if (!empty || own.length === 0) return;
        const visible = Array.from(own).some((el) => !el.hasAttribute("data-filter-hidden"));
        empty.hidden = visible;
      });

      if (count) count.textContent = tag ? `${shown} post${shown === 1 ? "" : "s"}` : "";

      if (updateUrl) {
        const url = new URL(window.location.href);
        if (tag) url.searchParams.set("tag", tag);
        else url.searchParams.delete("tag");
        window.history.replaceState({}, "", url);
      }
    };

    buttons.forEach((b) => b.addEventListener("click", () => apply(b.dataset.filterTag, true)));

    const initial = new URLSearchParams(window.location.search).get("tag");
    if (initial && buttons.some((b) => b.dataset.filterTag === initial.toLowerCase())) {
      apply(initial.toLowerCase(), false);
    }
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
      if (trap && (trap.checked || trap.value)) return;

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
