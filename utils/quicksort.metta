!(import! &self rand)

(= (car $x) (car-atom $x))
(= (cdr $x) (cdr-atom $x))
(= (cons $x $xs) (cons-atom $x $xs))

(= (zip $xs $ys)
   (if (or (== $xs ()) (== $ys ()))
       ()
       (cons (cons (car $xs) (cons (car $ys) ()))
             (zip (cdr $xs) (cdr $ys)))))

(= (tuple-first-less $a $b)
   (< (car-atom $a) (car-atom $b)))

(= (tuple-first-gte $a $b)
   (>= (car-atom $a) (car-atom $b)))

(= (filter-less $xs $pivot)
   (if (== $xs ()) ()
       (let* (( $x (car $xs) ) ($rest (cdr $xs)))
       (if (tuple-first-less $x $pivot)
           (cons $x (filter-less $rest $pivot))
           (filter-less $rest $pivot)))))

(= (filter-gte $xs $pivot)
   (if (== $xs ()) ()
       (let* (( $x (car $xs) ) ($rest (cdr $xs)))
       (if (tuple-first-gte $x $pivot)
           (cons $x (filter-gte $rest $pivot))
           (filter-gte $rest $pivot)))))

(= (append $xs $ys)
   (if (== $xs ()) $ys
       (cons (car $xs) (append (cdr $xs) $ys))))

(= (quick-sort $xs)
   (if (== $xs ()) ()
       (let* (
           ($pivot (car $xs))
           ($rest (cdr $xs))
           ($less (quick-sort (filter-less $rest $pivot)))
           ($greater (quick-sort (filter-gte $rest $pivot)))
       )
       (append $greater (cons $pivot $less)))))

; !(quick-sort (zip (randfloatarr! 5) ((0.1 0.2) (0.5 0.6) (0.1 0.7) (0.8 0.2) (0.9 0.2))))
!(zip (randfloatarr! 5) ((0.1 0.2) (0.5 0.6) (0.1 0.7) (0.8 0.2) (0.9 0.2)))



(= (slice $xs $n)
   (if (or (== $xs ()) (<= $n 0))
       ()
       (cons (car $xs) (slice (cdr $xs) (- $n 1)))))

!(slice (randfloatarr! 5) 3)