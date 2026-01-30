import type { Metadata } from "next";
import { Syne, Libre_Baskerville } from "next/font/google";
import "./globals.css";

const syne = Syne({
  variable: "--font-syne",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
});

const libre = Libre_Baskerville({
  variable: "--font-libre",
  subsets: ["latin"],
  weight: ["400", "700"],
  style: ["normal", "italic"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "CityLens — See Beyond the Surface",
  description: "Transform how you see urban spaces. AI-powered insights that reveal the stories, design, and meaning hidden in everyday city scenes.",
  keywords: ["urban exploration", "AI analysis", "city insights", "architecture", "travel"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${syne.variable} ${libre.variable} antialiased min-h-screen`}>
        {/* Noise texture overlay for that editorial feel */}
        <div className="noise-overlay" aria-hidden="true" />

        {/* Main content */}
        {children}
      </body>
    </html>
  );
}
