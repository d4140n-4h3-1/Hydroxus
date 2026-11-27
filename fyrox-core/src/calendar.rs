// Copyright (c) 2019-present Dmitry Stepanov and Fyrox Engine contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! A world-owned game clock with a fixed 365-day calendar.
//!
//! Store a GameCalendar in plugin state and pass it the simulation step in seconds
//! from PluginContext::dt. Use its fractional day of year for seasonal foliage
//! and its fractional hour for time-of-day lighting.
//!
//! ```
//! use fyrox_core::calendar::{CalendarDate, GameCalendar, Season};
//!
//! let date = CalendarDate::new(1, 10, 18)?;
//! // Start at 09:00 and advance one game minute per simulation second.
//! let mut clock = GameCalendar::new(date, 9.0 * 3600.0, 60.0)?;
//! clock.tick(1.0)?;
//! assert_eq!(clock.seconds_since_midnight(), 9.0 * 3600.0 + 60.0);
//! assert_eq!(clock.date().day_of_year(), 291);
//! assert_eq!(clock.date().season(), Season::Autumn);
//! # Ok::<(), fyrox_core::calendar::CalendarError>(())
//! ```

use std::fmt;

/// Days in one year. February always has 28 days.
pub const DAYS_PER_YEAR: u16 = 365;
/// Game seconds in one day.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

const MONTH_LENGTHS: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const TOTAL_DAYS: i64 = u32::MAX as i64 * DAYS_PER_YEAR as i64;

/// Northern Hemisphere meteorological seasons.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Season {
    /// March through May.
    Spring,
    /// June through August.
    Summer,
    /// September through November.
    Autumn,
    /// December through February.
    Winter,
}

/// Errors from creating or advancing a calendar.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CalendarError {
    /// The year, month, or day does not form a valid date.
    InvalidDate,
    /// The time of day, rate, or elapsed seconds are invalid.
    InvalidTime,
    /// The advance would take the calendar outside years 1 through u32::MAX.
    OutOfRange,
}

impl fmt::Display for CalendarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidDate => "invalid game calendar date",
            Self::InvalidTime => "invalid game calendar time or rate",
            Self::OutOfRange => "game calendar advance would exceed its year range",
        })
    }
}

impl std::error::Error for CalendarError {}

/// A validated date in a 365-day game year.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CalendarDate {
    year: u32,
    day_of_year: u16,
}

impl CalendarDate {
    /// Construct a date using one-based year, month, and day numbers.
    pub fn new(year: u32, month: u8, day: u8) -> Result<Self, CalendarError> {
        if !(1..=12).contains(&month) || day == 0 {
            return Err(CalendarError::InvalidDate);
        }
        let month_index = usize::from(month - 1);
        if day > MONTH_LENGTHS[month_index] {
            return Err(CalendarError::InvalidDate);
        }
        let previous_days: u16 = MONTH_LENGTHS[..month_index]
            .iter()
            .map(|&length| u16::from(length))
            .sum();
        Self::from_day_of_year(year, previous_days + u16::from(day))
    }

    /// Construct a date from a one-based year and day of year.
    pub fn from_day_of_year(year: u32, day_of_year: u16) -> Result<Self, CalendarError> {
        if year == 0 || !(1..=DAYS_PER_YEAR).contains(&day_of_year) {
            return Err(CalendarError::InvalidDate);
        }
        Ok(Self { year, day_of_year })
    }

    /// The one-based game year.
    pub fn year(self) -> u32 {
        self.year
    }

    /// The day in this year, from 1 (January 1) to 365 (December 31).
    pub fn day_of_year(self) -> u16 {
        self.day_of_year
    }

    /// The numeric month and day of month, both one-based.
    pub fn month_day(self) -> (u8, u8) {
        let mut day = self.day_of_year;
        for (index, length) in MONTH_LENGTHS.into_iter().enumerate() {
            if day <= u16::from(length) {
                return (index as u8 + 1, day as u8);
            }
            day -= u16::from(length);
        }
        unreachable!("validated calendar date")
    }

    /// Meteorological season for this date.
    pub fn season(self) -> Season {
        match self.day_of_year {
            60..=151 => Season::Spring,
            152..=243 => Season::Summer,
            244..=334 => Season::Autumn,
            _ => Season::Winter,
        }
    }
}

impl fmt::Display for CalendarDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (month, day) = self.month_day();
        write!(f, "{:04}-{month:02}-{day:02}", self.year)
    }
}

/// A simulation-driven date and time, independent of the host clock.
#[derive(Debug, Clone, PartialEq)]
pub struct GameCalendar {
    date: CalendarDate,
    seconds_since_midnight: f64,
    rate: f64,
    paused: bool,
}

impl GameCalendar {
    /// Start at the given date and time with a rate in game seconds per
    /// simulation second. The clock starts unpaused.
    pub fn new(
        date: CalendarDate,
        seconds_since_midnight: f64,
        rate: f64,
    ) -> Result<Self, CalendarError> {
        if !(0.0..SECONDS_PER_DAY).contains(&seconds_since_midnight) {
            return Err(CalendarError::InvalidTime);
        }
        let mut clock = Self {
            date,
            seconds_since_midnight,
            rate: 0.0,
            paused: false,
        };
        clock.set_rate(rate)?;
        Ok(clock)
    }

    /// The current date.
    pub fn date(&self) -> CalendarDate {
        self.date
    }

    /// The time of day in fractional seconds since midnight.
    pub fn seconds_since_midnight(&self) -> f64 {
        self.seconds_since_midnight
    }

    /// The time of day in fractional hours since midnight.
    pub fn hour(&self) -> f64 {
        self.seconds_since_midnight / 3600.0
    }

    /// The one-based day of year with time of day included as a fraction.
    pub fn fractional_day_of_year(&self) -> f64 {
        f64::from(self.date.day_of_year) + self.seconds_since_midnight / SECONDS_PER_DAY
    }

    /// Game seconds advanced per simulation second.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Change the time scale. Zero holds the current date and time.
    pub fn set_rate(&mut self, rate: f64) -> Result<(), CalendarError> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(CalendarError::InvalidTime);
        }
        self.rate = rate;
        Ok(())
    }

    /// Whether simulation steps are currently ignored.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Pause or resume advancing with simulation steps.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Advance by a nonnegative, finite simulation step in seconds.
    ///
    /// Returns true if the date changed. A failed step leaves the clock intact.
    pub fn tick(&mut self, simulation_seconds: f64) -> Result<bool, CalendarError> {
        if !simulation_seconds.is_finite() || simulation_seconds < 0.0 {
            return Err(CalendarError::InvalidTime);
        }
        if self.paused {
            return Ok(false);
        }
        self.advance_game_seconds(simulation_seconds * self.rate)
    }

    /// Jump by a signed amount of game time, independently of pause and rate.
    ///
    /// Returns true if the date changed. A failed jump leaves the clock intact.
    pub fn advance_game_seconds(&mut self, seconds: f64) -> Result<bool, CalendarError> {
        let total = self.seconds_since_midnight + seconds;
        if !total.is_finite() {
            return Err(CalendarError::InvalidTime);
        }
        let mut days = total.div_euclid(SECONDS_PER_DAY);
        let mut remainder = total.rem_euclid(SECONDS_PER_DAY);
        // A negative remainder very close to zero can round up to a full day.
        if remainder >= SECONDS_PER_DAY {
            days += 1.0;
            remainder = 0.0;
        }
        if days.abs() >= TOTAL_DAYS as f64 {
            return Err(CalendarError::OutOfRange);
        }
        let year_length = i64::from(DAYS_PER_YEAR);
        let old_day =
            i64::from(self.date.year - 1) * year_length + i64::from(self.date.day_of_year - 1);
        let new_day = old_day + days as i64;
        if !(0..TOTAL_DAYS).contains(&new_day) {
            return Err(CalendarError::OutOfRange);
        }
        self.date = CalendarDate {
            year: (new_day / year_length) as u32 + 1,
            day_of_year: (new_day % year_length) as u16 + 1,
        };
        self.seconds_since_midnight = remainder;
        Ok(new_day != old_day)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_roundtrip_and_seasons_match_foliage_days() {
        for ordinal in 1..=DAYS_PER_YEAR {
            let date = CalendarDate::from_day_of_year(2, ordinal).unwrap();
            let (month, day) = date.month_day();
            assert_eq!(CalendarDate::new(2, month, day), Ok(date));
            assert_eq!(
                date.season(),
                match month {
                    3..=5 => Season::Spring,
                    6..=8 => Season::Summer,
                    9..=11 => Season::Autumn,
                    _ => Season::Winter,
                }
            );
        }
        assert_eq!(CalendarDate::new(1, 10, 18).unwrap().day_of_year(), 291);
        assert_eq!(CalendarDate::new(1, 12, 1).unwrap().day_of_year(), 335);
        assert_eq!(CalendarDate::new(1, 2, 29), Err(CalendarError::InvalidDate));
        assert_eq!(CalendarDate::new(0, 1, 1), Err(CalendarError::InvalidDate));
    }

    #[test]
    fn ticks_and_jumps_cross_midnight_and_years() {
        let date = CalendarDate::new(7, 12, 31).unwrap();
        let mut clock = GameCalendar::new(date, SECONDS_PER_DAY - 10.0, 60.0).unwrap();
        assert!(clock.tick(1.0).unwrap());
        assert_eq!(clock.date(), CalendarDate::new(8, 1, 1).unwrap());
        assert_eq!(clock.seconds_since_midnight(), 50.0);
        assert!(clock.advance_game_seconds(-60.0).unwrap());
        assert_eq!(clock.date(), date);
        assert_eq!(clock.seconds_since_midnight(), SECONDS_PER_DAY - 10.0);
        assert!(clock.advance_game_seconds(730.0 * SECONDS_PER_DAY).unwrap());
        assert_eq!(clock.date(), CalendarDate::new(9, 12, 31).unwrap());
    }

    #[test]
    fn invalid_steps_preserve_state_and_pause_allows_manual_jumps() {
        let date = CalendarDate::new(1, 1, 1).unwrap();
        let mut clock = GameCalendar::new(date, 0.0, 60.0).unwrap();
        let original = clock.clone();
        assert_eq!(
            clock.advance_game_seconds(-1.0),
            Err(CalendarError::OutOfRange)
        );
        assert_eq!(clock.tick(f64::NAN), Err(CalendarError::InvalidTime));
        assert_eq!(clock.tick(f64::MAX), Err(CalendarError::InvalidTime));
        assert_eq!(clock.set_rate(-1.0), Err(CalendarError::InvalidTime));
        assert_eq!(clock, original);
        clock.set_paused(true);
        assert!(!clock.tick(2000.0).unwrap());
        assert_eq!(clock.date(), date);
        assert!(clock.advance_game_seconds(SECONDS_PER_DAY * 1.5).unwrap());
        assert_eq!(clock.fractional_day_of_year(), 2.5);
        assert_eq!(clock.hour(), 12.0);
    }
}
