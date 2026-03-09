/**
 * FraudGuard AI — Precision Scroll Animation Engine
 * scroll.js
 *
 * Techniques used:
 *  • Lerp (linear interpolation) for silky parallax
 *  • IntersectionObserver for morph triggers
 *  • Scroll-position-linked CSS custom props
 *  • Character split text reveal
 *  • Magnetic button effect
 *  • Timeline fill synchronized with scroll
 *  • Bento card mouse-tracking radial glow
 */

'use strict';

/* ─── Utility ─── */
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, min, max) => Math.min(Math.max(v, min), max);
const map = (v, inMin, inMax, outMin, outMax) =>
    outMin + ((v - inMin) / (inMax - inMin)) * (outMax - outMin);
const qs = (sel, ctx = document) => ctx.querySelector(sel);
const qsa = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

/* ─── Cursor (smooth via lerp) ─── */
(function initCursor() {
    const cursor = qs('.cursor');
    const dot = qs('.cursor-dot');
    if (!cursor || !dot) return;

    let mx = -500, my = -500;  // target (mouse)
    let cx = -500, cy = -500;  // current ring (lerped)

    document.addEventListener('mousemove', e => {
        mx = e.clientX;
        my = e.clientY;
        // dot follows cursor instantly via transform (zero lag)
        dot.style.transform = `translate(calc(${mx}px - 50%), calc(${my}px - 50%))`;
    }, { passive: true });

    function animCursor() {
        cx = lerp(cx, mx, 0.18);   // 0.18 = snappy but still smooth
        cy = lerp(cy, my, 0.18);
        cursor.style.transform = `translate(calc(${cx}px - 50%), calc(${cy}px - 50%))`;
        requestAnimationFrame(animCursor);
    }
    requestAnimationFrame(animCursor);

    // Hovering interactive elements
    const hoverEls = qsa('a, button, .bento-card, .tl-item, .step-card');
    hoverEls.forEach(el => {
        el.addEventListener('mouseenter', () => cursor.classList.add('hovering'));
        el.addEventListener('mouseleave', () => cursor.classList.remove('hovering'));
    });
})();

/* ─── Scroll Progress Bar ─── */
(function initScrollBar() {
    const bar = qs('#scrollBar');
    if (!bar) return;
    window.addEventListener('scroll', () => {
        const pct = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
        bar.style.width = clamp(pct, 0, 100) + '%';
    }, { passive: true });
})();

/* ─── Sticky / scrolled header ─── */
(function initHeader() {
    const header = qs('#siteHeader');
    if (!header) return;
    window.addEventListener('scroll', () => {
        header.classList.toggle('scrolled', window.scrollY > 50);
    }, { passive: true });
})();

/* ─── Smooth scroll for anchor links ─── */
qsa('a.smooth-scroll, a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
        const href = a.getAttribute('href');
        if (!href.startsWith('#')) return;
        e.preventDefault();
        const target = qs(href);
        if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
});

/* ─── Mobile nav ─── */
(function initMobileNav() {
    const toggle = qs('#navToggle');
    const menu = qs('#navMenu');
    if (!toggle || !menu) return;
    toggle.addEventListener('click', () => {
        menu.classList.toggle('open');
        toggle.classList.toggle('open');
    });
})();

/* ─── Hero entrance (simple, reliable) ─── */
(function initHeroEntrance() {
    const tag = qs('.hero-tag');
    const h1 = qs('#heroH1');
    const sub = qs('#heroSub');
    const btns = qs('#heroBtns');
    const shield = qs('#heroShield');

    // Simple stagger — just add a class to let CSS transition kick in
    function show(el, delay) {
        if (!el) return;
        setTimeout(() => el.classList.add('show'), delay);
    }

    requestAnimationFrame(() => {
        show(tag, 200);
        show(h1, 350);
        show(sub, 550);
        show(btns, 720);
        show(shield, 500);
    });

    // Chips stagger
    qsa('.d-chip').forEach((chip, i) => {
        setTimeout(() => chip.classList.add('show'), 850 + i * 220);
    });
})();

/* ─── Parallax: lerp-driven ─── */
(function initParallax() {
    const layers = qsa('[data-depth]');
    if (!layers.length) return;

    let scrollY = 0;
    let targetY = 0;
    let shieldTarget = 0;
    let shieldCurrent = 0;

    const shield = qs('#heroShield');

    window.addEventListener('scroll', () => {
        targetY = window.scrollY;
    }, { passive: true });

    function tick() {
        scrollY = lerp(scrollY, targetY, 0.08); // smooth lerp

        layers.forEach(layer => {
            const depth = parseFloat(layer.dataset.depth) || 0;
            layer.style.transform = `translateY(${scrollY * depth}px)`;
        });

        // Shield sinks a bit faster — gives depth
        if (shield) {
            shieldCurrent = lerp(shieldCurrent, targetY * 0.12, 0.08);
            shield.style.transform = `translateY(${shieldCurrent}px)`;
        }

        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
})();

/* ─── Morph-in: scroll reveal with clip-path ─── */
(function initMorphReveal() {
    const els = qsa('.morph-in');
    if (!els.length) return;

    const obs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const delay = parseInt(el.dataset.morphDelay || 0, 10);
            setTimeout(() => el.classList.add('is-visible'), delay);
            obs.unobserve(el);
        });
    }, {
        threshold: 0.08,
        rootMargin: '0px 0px -40px 0px'
    });

    els.forEach(el => obs.observe(el));
})();

/* ─── Timeline fill synchronized with scroll ─── */
(function initTimelineFill() {
    const section = qs('.s-how');
    const bar = qs('#tlProgress');
    if (!section || !bar) return;

    window.addEventListener('scroll', () => {
        const rect = section.getBoundingClientRect();
        const winH = window.innerHeight;
        const visible = clamp(map(rect.top, winH, -rect.height, 0, 1), 0, 1);
        bar.style.height = (visible * 100) + '%';
    }, { passive: true });
})();

/* ─── Count-up on scroll ─── */
(function initCountUp() {
    const els = qsa('.count-up');
    if (!els.length) return;

    const obs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseFloat(el.dataset.target);
            const suffix = el.dataset.suffix || '';
            const isFloat = target % 1 !== 0;
            const dur = 1600;
            const start = performance.now();

            el.classList.add('counted');

            function step(now) {
                const p = clamp((now - start) / dur, 0, 1);
                const ease = 1 - Math.pow(1 - p, 4);
                const val = target * ease;
                el.textContent = (isFloat ? val.toFixed(1) : Math.round(val)) + suffix;
                if (p < 1) requestAnimationFrame(step);
            }
            requestAnimationFrame(step);
            obs.unobserve(el);
        });
    }, { threshold: 0.5 });

    els.forEach(el => obs.observe(el));
})();

/* ─── Bento card: mouse tracking radial glow ─── */
(function initBentoGlow() {
    qsa('.bento-card').forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width * 100).toFixed(1);
            const y = ((e.clientY - rect.top) / rect.height * 100).toFixed(1);
            card.style.setProperty('--mx', x + '%');
            card.style.setProperty('--my', y + '%');
        });
    });
})();

/* ─── Step/Timeline card hover lift (subtle tilt) ─── */
(function initTiltCards() {
    qsa('.tl-item, .step-card').forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            card.style.transform = `translateY(-6px) rotateX(${(-y * 6).toFixed(2)}deg) rotateY(${(x * 6).toFixed(2)}deg)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
})();
