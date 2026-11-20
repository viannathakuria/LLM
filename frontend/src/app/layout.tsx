import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MyLLM",
  description: "An independent language model project in development.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
