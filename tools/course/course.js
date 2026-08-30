/* ============================================================
   Shared chrome for every LLM-course lesson page.
   - top progress bar reflects scroll depth
   - .reveal elements fade in on scroll
   - .check quizzes: click an .opt to grade + reveal .why
   - ArrowLeft / ArrowRight jump to prev / next lesson
   Include with: <script src="course.js" defer></script>
   ============================================================ */
(function(){
  "use strict";
  const $  = (s,r)=> (r||document).querySelector(s);
  const $$ = (s,r)=> Array.from((r||document).querySelectorAll(s));

  /* ---- top progress bar ---- */
  const fill = $('.topbar .fill');
  if(fill){
    const onScroll = ()=>{
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      fill.style.width = (max>0 ? (h.scrollTop/max*100) : 0) + '%';
    };
    document.addEventListener('scroll', onScroll, {passive:true});
    window.addEventListener('resize', onScroll, {passive:true});
    onScroll();
  }

  /* ---- scroll reveal ---- */
  if('IntersectionObserver' in window){
    const revObs = new IntersectionObserver((es)=>{
      es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); revObs.unobserve(e.target); } });
    },{threshold:.12});
    $$('.reveal').forEach(el=>revObs.observe(el));
  } else {
    $$('.reveal').forEach(el=>el.classList.add('in'));
  }

  /* ---- self-check quizzes ---- */
  $$('.check').forEach(box=>{
    const why = $('.why', box);
    let answered = false;
    $$('.opt', box).forEach(opt=>{
      opt.addEventListener('click', ()=>{
        if(answered) return;
        answered = true;
        const correct = opt.dataset.correct === '1';
        opt.classList.add(correct ? 'correct' : 'wrong');
        if(!correct){
          const right = $$('.opt', box).find(o=>o.dataset.correct==='1');
          if(right) right.classList.add('correct');
        }
        if(why) why.classList.add('show');
      });
    });
  });

  /* ---- keyboard prev / next ---- */
  document.addEventListener('keydown', (e)=>{
    if(e.target.matches('input,textarea,select')) return;
    if(e.key === 'ArrowRight'){ const n = $('a[data-nav="next"]'); if(n && !n.classList.contains('disabled')) location.href = n.href; }
    if(e.key === 'ArrowLeft'){  const p = $('a[data-nav="prev"]'); if(p && !p.classList.contains('disabled')) location.href = p.href; }
  });
})();
