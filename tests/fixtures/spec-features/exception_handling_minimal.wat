;; Exception handling minimal fixture (post-MVP wasm feature)
;; LOOM does not currently support EH; parser must reject cleanly (no panic).
;; Uses the legacy try/catch encoding plus throw/throw_ref for breadth.
(module
  (tag $exn (param i32))
  (func (export "eh_test") (result i32)
    (try (result i32)
      (do
        i32.const 1
        throw $exn
        i32.const 0
      )
      (catch $exn
        ;; on catch the i32 payload is on the stack
      )
      (catch_all
        i32.const -1
      )
    )
  )
)
