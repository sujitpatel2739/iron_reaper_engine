class Observer:
    def on_forward_pre(self, ctx: ExecutionContext): pass
    def on_forward_post(self, ctx: ExecutionContext): pass
    def on_backward_pre(self, ctx: ExecutionContext): pass
    def on_backward_post(self, ctx: ExecutionContext): pass

    def finalize(self): pass
