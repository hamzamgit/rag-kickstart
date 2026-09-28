-- This is safe for a database that already has the original documents table.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER;
