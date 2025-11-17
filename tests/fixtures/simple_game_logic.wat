;; Simple game logic example
;; Shows realistic patterns from game development

(module
  (memory 1)

  ;; Calculate damage with various modifiers
  (func $calculate_damage (export "calc_damage")
    (param $base_damage i32)
    (param $attack_power i32)
    (param $defense i32)
    (param $crit_chance i32)
    (result i32)
    (local $damage i32)
    (local $multiplier i32)

    ;; Start with base damage
    (local.set $damage (local.get $base_damage))

    ;; Add attack power bonus
    ;; damage += attack_power * 2 (opportunity for strength reduction)
    (local.set $damage
      (i32.add
        (local.get $damage)
        (i32.mul (local.get $attack_power) (i32.const 2))
      )
    )

    ;; Apply defense reduction
    ;; damage -= defense / 4 (opportunity for strength reduction)
    (local.set $damage
      (i32.sub
        (local.get $damage)
        (i32.div_u (local.get $defense) (i32.const 4))
      )
    )

    ;; Critical hit check (simplified)
    ;; If crit_chance > 50, double damage
    (if (i32.gt_u (local.get $crit_chance) (i32.const 50))
      (then
        ;; damage *= 2 (strength reduction opportunity)
        (local.set $damage
          (i32.mul (local.get $damage) (i32.const 2))
        )
      )
    )

    ;; Ensure damage is at least 1
    (if (result i32) (i32.le_s (local.get $damage) (i32.const 0))
      (then (i32.const 1))
      (else (local.get $damage))
    )
  )

  ;; Update player position with bounds checking
  (func $update_position (export "update_pos")
    (param $x i32)
    (param $y i32)
    (param $dx i32)
    (param $dy i32)
    (result i32 i32)
    (local $new_x i32)
    (local $new_y i32)
    (local $max_x i32)
    (local $max_y i32)

    ;; Constants (opportunities for constant propagation)
    (local.set $max_x (i32.const 1920))
    (local.set $max_y (i32.const 1080))

    ;; Calculate new position
    (local.set $new_x (i32.add (local.get $x) (local.get $dx)))
    (local.set $new_y (i32.add (local.get $y) (local.get $dy)))

    ;; Clamp X to bounds
    (if (i32.lt_s (local.get $new_x) (i32.const 0))
      (then (local.set $new_x (i32.const 0)))
    )
    (if (i32.gt_u (local.get $new_x) (local.get $max_x))
      (then (local.set $new_x (local.get $max_x)))
    )

    ;; Clamp Y to bounds
    (if (i32.lt_s (local.get $new_y) (i32.const 0))
      (then (local.set $new_y (i32.const 0)))
    )
    (if (i32.gt_u (local.get $new_y) (local.get $max_y))
      (then (local.set $new_y (local.get $max_y)))
    )

    ;; Return new position (multi-value)
    (local.get $new_x)
    (local.get $new_y)
  )

  ;; Calculate distance squared (no sqrt for performance)
  (func $distance_squared (export "dist_sq")
    (param $x1 i32) (param $y1 i32)
    (param $x2 i32) (param $y2 i32)
    (result i32)
    (local $dx i32)
    (local $dy i32)

    ;; dx = x2 - x1
    (local.set $dx (i32.sub (local.get $x2) (local.get $x1)))

    ;; dy = y2 - y1
    (local.set $dy (i32.sub (local.get $y2) (local.get $y1)))

    ;; return dx*dx + dy*dy
    (i32.add
      (i32.mul (local.get $dx) (local.get $dx))
      (i32.mul (local.get $dy) (local.get $dy))
    )
  )
)
