//! A cluster graph as bytes, to build it once per mesh and keep it next to the model.
//!
//! Little-endian throughout: the magic `YOCT`, a version, then the clusters and the groups,
//! each list preceded by its length.

use crate::{Cluster, ClusterDag, Group, Sphere};

const MAGIC: &[u8; 4] = b"YOCT";
const VERSION: u32 = 1;

/// Why bytes are not a cluster graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// They do not start as one does.
    NotYoctite,
    /// Made by another version of the format, to be built again.
    Version(u32),
    /// They end before it does, or say something no cluster graph says.
    Corrupt,
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotYoctite => write!(f, "not a Yoctite cluster graph"),
            Self::Version(v) => write!(f, "Yoctite cluster graph version {v}, expected {VERSION}"),
            Self::Corrupt => write!(f, "corrupt Yoctite cluster graph"),
        }
    }
}

impl std::error::Error for FormatError {}

const NONE: u32 = u32::MAX;

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn sphere(&mut self, s: &Sphere) {
        for v in s.center {
            self.f32(v);
        }
        self.f32(s.radius);
    }
    fn list(&mut self, items: &[u32]) {
        self.u32(items.len() as u32);
        for &i in items {
            self.u32(i);
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], FormatError> {
        if self.0.len() < n {
            return Err(FormatError::Corrupt);
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, FormatError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn sphere(&mut self) -> Result<Sphere, FormatError> {
        Ok(Sphere {
            center: [self.f32()?, self.f32()?, self.f32()?],
            radius: self.f32()?,
        })
    }
    /// A length, no longer than what is left could hold at `each` bytes an item.
    fn len(&mut self, each: usize) -> Result<usize, FormatError> {
        let n = self.u32()? as usize;
        if n.saturating_mul(each) > self.0.len() {
            return Err(FormatError::Corrupt);
        }
        Ok(n)
    }
    fn list(&mut self) -> Result<Vec<u32>, FormatError> {
        let n = self.len(4)?;
        (0..n).map(|_| self.u32()).collect()
    }
    fn option(&mut self) -> Result<Option<u32>, FormatError> {
        let v = self.u32()?;
        Ok((v != NONE).then_some(v))
    }
}

impl ClusterDag {
    /// It as bytes, for [`Self::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        w.0.extend_from_slice(MAGIC);
        w.u32(VERSION);
        w.u32(self.clusters.len() as u32);
        for c in &self.clusters {
            w.list(&c.vertices);
            w.u32(c.triangles.len() as u32);
            for t in &c.triangles {
                w.0.extend_from_slice(t);
            }
            w.sphere(&c.bounds);
            w.f32(c.error);
            w.sphere(&c.lod_bounds);
            w.f32(c.parent_error);
            w.sphere(&c.parent_bounds);
            w.u32(c.level);
            w.u32(c.made_from.unwrap_or(NONE));
            w.u32(c.simplified_in.unwrap_or(NONE));
        }
        w.u32(self.groups.len() as u32);
        for g in &self.groups {
            w.list(&g.children);
            w.list(&g.parents);
            w.f32(g.error);
            w.sphere(&g.bounds);
            w.u32(g.level);
        }
        w.0
    }

    /// A cluster graph from [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader(bytes);
        if r.take(4).map_err(|_| FormatError::NotYoctite)? != MAGIC {
            return Err(FormatError::NotYoctite);
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(FormatError::Version(version));
        }
        let cluster_count = r.len(4)?;
        let mut clusters = Vec::with_capacity(cluster_count);
        for _ in 0..cluster_count {
            let vertices = r.list()?;
            let triangle_count = r.len(3)?;
            let triangles = (0..triangle_count)
                .map(|_| {
                    let t: [u8; 3] = r.take(3)?.try_into().unwrap();
                    if t.iter().any(|&i| i as usize >= vertices.len()) {
                        return Err(FormatError::Corrupt);
                    }
                    Ok(t)
                })
                .collect::<Result<_, _>>()?;
            clusters.push(Cluster {
                vertices,
                triangles,
                bounds: r.sphere()?,
                error: r.f32()?,
                lod_bounds: r.sphere()?,
                parent_error: r.f32()?,
                parent_bounds: r.sphere()?,
                level: r.u32()?,
                made_from: r.option()?,
                simplified_in: r.option()?,
            });
        }
        let group_count = r.len(4)?;
        let mut groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            groups.push(Group {
                children: r.list()?,
                parents: r.list()?,
                error: r.f32()?,
                bounds: r.sphere()?,
                level: r.u32()?,
            });
        }
        if !r.0.is_empty() {
            return Err(FormatError::Corrupt);
        }
        let in_range = |i: &u32, n: usize| (*i as usize) < n;
        let consistent = clusters.iter().all(|c| {
            c.made_from.is_none_or(|g| in_range(&g, groups.len()))
                && c.simplified_in.is_none_or(|g| in_range(&g, groups.len()))
        }) && groups.iter().all(|g| {
            g.children.iter().chain(&g.parents).all(|c| in_range(c, clusters.len()))
        });
        if !consistent {
            return Err(FormatError::Corrupt);
        }
        Ok(Self { clusters, groups })
    }
}
