import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-md border-[#ff8c00] px-2 py-0.25 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 whitespace-nowrap h-5",
  {
    variants: {
      variant: {
        default:
          "border-[#ff8c00] bg-primary text-primary-foreground shadow hover:bg-primary/80",
        secondary:
          "border-[#ff8c00] bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-[#ff8c00] bg-destructive text-destructive-foreground shadow hover:bg-destructive/80",
        outline: "text-foreground border-[#ff8c00]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
