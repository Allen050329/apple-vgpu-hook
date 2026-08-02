-- reposition.applescript — move a window along a path as fast as the
-- accessibility API allows, for a fixed number of steps.
--
-- This is the fallback motion, and it is not a pointer drag. A real drag needs
-- CGEventPost (see drag.c), whose events this guest silently discards because
-- the posting process is not trusted for Accessibility and TCC.db cannot be
-- written here: there is no passwordless sudo and SIP's Filesystem Protections
-- are on. System Events *is* trusted, so this route moves the window while the
-- other one does not.
--
-- What that costs is stated rather than glossed: the window server sees a
-- sequence of window moves instead of a pointer held across a title bar, so any
-- work specific to a drag session is missing. What it keeps is the part the
-- device sees — a large window changing position at high rate, and everything
-- behind it recomposited.
--
-- Returns the number of steps performed; the caller times the run, because
-- AppleScript's `current date` has one-second resolution and cannot.
on run argv
	set appName to item 1 of argv
	set x0 to (item 2 of argv) as integer
	set y0 to (item 3 of argv) as integer
	set steps to (item 4 of argv) as integer
	set ampX to (item 5 of argv) as integer
	set ampY to (item 6 of argv) as integer

	set done to 0
	tell application "System Events" to tell process appName
		repeat with i from 1 to steps
			-- Two incommensurate periods, so the path is not a straight slide,
			-- which is the one motion a compositor may coalesce. Integer
			-- arithmetic only: AppleScript has no cheap sine and the point is
			-- the motion, not its exact shape.
			set dx to ((i * 7) mod (2 * ampX)) - ampX
			set dy to ((i * 11) mod (2 * ampY)) - ampY
			set position of window 1 to {x0 + dx, y0 + dy}
			set done to done + 1
		end repeat
	end tell
	return done
end run
