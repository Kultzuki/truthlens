"use client";

import Link from 'next/link';
import { Github, Linkedin, Twitter } from 'lucide-react';
import { cn } from '@/lib/utils';

export function Footer() {
  return (
    <footer className="relative overflow-hidden rounded-t-3xl bg-primary/95 backdrop-blur-sm border-t border-white/10 py-12 sm:py-16 shadow-[0_-10px_40px_rgba(0,0,0,0.35)]">
      {/* Background radial gradient */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <svg className="absolute -top-32 -left-24 opacity-20" width="540" height="540" viewBox="0 0 540 540">
          <defs>
            <radialGradient id="footer-radial-a" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="white" stopOpacity="0.35" />
              <stop offset="100%" stopColor="white" stopOpacity="0" />
            </radialGradient>
          </defs>
          <circle cx="270" cy="270" r="270" fill="url(#footer-radial-a)" />
        </svg>
        <svg className="absolute -bottom-40 -right-32 opacity-15" width="640" height="640" viewBox="0 0 640 640">
          <defs>
            <radialGradient id="footer-radial-b" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="white" stopOpacity="0.25" />
              <stop offset="100%" stopColor="white" stopOpacity="0" />
            </radialGradient>
          </defs>
          <circle cx="320" cy="320" r="320" fill="url(#footer-radial-b)" />
        </svg>
      </div>

      {/* Top glow line */}
      <div aria-hidden className="absolute -top-px left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/40 to-transparent" />
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-12 md:grid-cols-12">
          {/* Section 1: Logo & Tagline */}
          <div className="md:col-span-4 lg:col-span-5">
            <Link href="/" className="text-2xl font-bold text-glow tracking-wider text-white hover:text-white/90 transition-colors">
              TRUTHLENS.AI
            </Link>
            <p className="mt-4 text-sm text-white/90">
              Making Truth Verifiable, One Frame at a Time.
            </p>
          </div>

          {/* Section 2: Navigation */}
          <div className="md:col-span-4 lg:col-span-3">
            <h3 className="text-sm font-semibold text-white inline-flex items-center gap-2">
              <span>Navigation</span>
              <span className="h-px w-10 bg-white/40" />
            </h3>
            <ul className="mt-4 space-y-2">
              <li><Link href="#" className="text-sm text-white/85 hover:text-white transition-colors">About Us</Link></li>
              <li><Link href="#" className="text-sm text-white/85 hover:text-white transition-colors">Contact</Link></li>
              <li><Link href="#" className="text-sm text-white/85 hover:text-white transition-colors">GitHub Repo</Link></li>
            </ul>
          </div>

          {/* Section 3: Legal & Social */}
          <div className="md:col-span-4 lg:col-span-4">
            <h3 className="text-sm font-semibold text-white inline-flex items-center gap-2">
              <span>Connect With Us</span>
              <span className="h-px w-10 bg-white/40" />
            </h3>
            <div className="mt-4 flex space-x-4">
              <Link href="#" className="text-white/85 hover:text-white transition-transform hover:scale-110">
                <Twitter className="h-6 w-6" />
                <span className="sr-only">Twitter</span>
              </Link>
              <Link href="#" className="text-white/85 hover:text-white transition-transform hover:scale-110">
                <Github className="h-6 w-6" />
                <span className="sr-only">GitHub</span>
              </Link>
              <Link href="#" className="text-white/85 hover:text-white transition-transform hover:scale-110">
                <Linkedin className="h-6 w-6" />
                <span className="sr-only">LinkedIn</span>
              </Link>
            </div>
          </div>
        </div>
         <div className="mt-12 border-t border-white/20 pt-8 text-center text-sm text-white/85">
           &copy; {new Date().getFullYear()} TRUTHLENS.AI. All rights reserved.
         </div>
      </div>
    </footer>
  );
}
