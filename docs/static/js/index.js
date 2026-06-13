(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {

    /* ---------------- Mobile navbar burger ---------------- */
    var burger = document.querySelector('.navbar-burger');
    if (burger) {
      burger.addEventListener('click', function () {
        var target = document.getElementById(burger.dataset.target);
        burger.classList.toggle('is-active');
        if (target) target.classList.toggle('is-active');
      });
    }

    /* ---------------- Smooth-scroll nav (close mobile menu on click) ---------------- */
    document.querySelectorAll('a.nav-link, a.navbar-title').forEach(function (link) {
      link.addEventListener('click', function () {
        var menu = document.getElementById('mainNav');
        if (menu && menu.classList.contains('is-active')) {
          menu.classList.remove('is-active');
          if (burger) burger.classList.remove('is-active');
        }
      });
    });

    /* ---------------- Gallery tab switching ---------------- */
    var tabItems = document.querySelectorAll('.gallery-tabs li');
    tabItems.forEach(function (item) {
      item.addEventListener('click', function () {
        var targetId = item.dataset.tab;
        tabItems.forEach(function (i) { i.classList.remove('is-active'); });
        item.classList.add('is-active');
        document.querySelectorAll('.tab-content').forEach(function (c) {
          var active = c.id === targetId;
          c.classList.toggle('is-active', active);
          // play videos in the active tab, pause the rest
          c.querySelectorAll('video').forEach(function (v) {
            if (active) { safePlay(v); } else { v.pause(); }
          });
        });
      });
    });

    /* ---------------- Click to toggle native controls ---------------- */
    function safePlay(v) {
      var p = v.play();
      if (p && typeof p.catch === 'function') { p.catch(function () {}); }
    }
    document.querySelectorAll('video').forEach(function (v) {
      v.addEventListener('click', function () {
        v.controls = !v.controls;
      });
    });

    /* ---------------- Video carousel ---------------- */
    document.querySelectorAll('[data-carousel]').forEach(function (carousel) {
      var track = carousel.querySelector('.carousel-track');
      var slides = Array.prototype.slice.call(carousel.querySelectorAll('.carousel-slide'));
      var prevBtn = carousel.querySelector('.carousel-prev');
      var nextBtn = carousel.querySelector('.carousel-next');
      var dotsContainer = carousel.querySelector('.carousel-dots');
      var index = 0;

      var dots = slides.map(function (_, i) {
        var dot = document.createElement('button');
        dot.className = 'carousel-dot' + (i === 0 ? ' is-active' : '');
        dot.setAttribute('aria-label', 'slide ' + (i + 1));
        dot.addEventListener('click', function () { goTo(i); });
        if (dotsContainer) dotsContainer.appendChild(dot);
        return dot;
      });

      function update() {
        track.style.transform = 'translateX(' + (-index * 100) + '%)';
        dots.forEach(function (d, i) { d.classList.toggle('is-active', i === index); });
        // play the visible slide's video, pause the others
        slides.forEach(function (s, i) {
          var v = s.querySelector('video');
          if (!v) return;
          if (i === index) { safePlay(v); } else { v.pause(); }
        });
      }

      function goTo(i) {
        index = (i + slides.length) % slides.length;
        update();
      }

      if (prevBtn) prevBtn.addEventListener('click', function () { goTo(index - 1); });
      if (nextBtn) nextBtn.addEventListener('click', function () { goTo(index + 1); });
    });

  });
})();
